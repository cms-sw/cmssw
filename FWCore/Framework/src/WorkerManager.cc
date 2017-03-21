#include "FWCore/Framework/interface/WorkerManager.h"
#include "UnscheduledConfigurator.h"

#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/ExceptionCollector.h"
#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"

static const std::string kFilterType("EDFilter");
static const std::string kProducerType("EDProducer");

namespace edm {
  // -----------------------------

  WorkerManager::WorkerManager(std::shared_ptr<ActivityRegistry> areg, ExceptionToActionTable const& actions) :
    workerReg_(areg),
    actionTable_(&actions),
    allWorkers_(),
    unscheduled_(*areg),
    lastSetupEventPrincipal_(nullptr)
  {

  } // WorkerManager::WorkerManager

  WorkerManager::WorkerManager(std::shared_ptr<ModuleRegistry> modReg,
                               std::shared_ptr<ActivityRegistry> areg,
                               ExceptionToActionTable const& actions) :
  workerReg_(areg,modReg),
  actionTable_(&actions),
  allWorkers_(),
  unscheduled_(*areg),
  lastSetupEventPrincipal_(nullptr)
  {
  } // WorkerManager::WorkerManager

  Worker* WorkerManager::getWorker(ParameterSet& pset,
                                   ProductRegistry& preg,
                                   PreallocationConfiguration const* prealloc,
                                   std::shared_ptr<ProcessConfiguration const> processConfiguration,
                                   std::string const & label) {
    WorkerParams params(&pset, preg, prealloc, processConfiguration, *actionTable_);
    return workerReg_.getWorker(params, label);
  }

  void WorkerManager::addToUnscheduledWorkers(ParameterSet& pset,
                                              ProductRegistry& preg,
                                              PreallocationConfiguration const* prealloc,
                                              std::shared_ptr<ProcessConfiguration> processConfiguration,
                                              std::string label,
                                              std::set<std::string>& unscheduledLabels,
                                              std::vector<std::string>& shouldBeUsedLabels) {
    //Need to
    // 1) create worker
    // 2) if it is a WorkerT<EDProducer>, add it to our list
    auto modType = pset.getParameter<std::string>("@module_edm_type");
    if(modType == kProducerType || modType == kFilterType) {
      Worker* newWorker = getWorker(pset, preg, prealloc, processConfiguration, label);
      assert(newWorker->moduleType() == Worker::kProducer || newWorker->moduleType() == Worker::kFilter);
      unscheduledLabels.insert(label);
      unscheduled_.addWorker(newWorker);
      //add to list so it gets reset each new event
      addToAllWorkers(newWorker);
    } else {
      shouldBeUsedLabels.push_back(label);
    }
  }

  void WorkerManager::setOnDemandProducts(ProductRegistry& pregistry, std::set<std::string> const& unscheduledLabels) const {
    for(auto& prod : pregistry.productListUpdator()) {
      if(prod.second.produced() &&
          prod.second.branchType() == InEvent &&
          unscheduledLabels.end() != unscheduledLabels.find(prod.second.moduleLabel())) {
        prod.second.setOnDemand(true);
      }
    }
  }
  
  void WorkerManager::endJob() {
    for(auto& worker : allWorkers_) {
      worker->endJob();
    }
  }

  void WorkerManager::endJob(ExceptionCollector& collector) {
    for(auto& worker : allWorkers_) {
      try {
        convertException::wrap([&]() {
          worker->endJob();
        });
      }
      catch (cms::Exception const& ex) {
        collector.addException(ex);
      }
    }
  }


  void WorkerManager::beginJob(ProductRegistry const& iRegistry) {
    auto const runLookup = iRegistry.productLookup(InRun);
    auto const lumiLookup = iRegistry.productLookup(InLumi);
    auto const eventLookup = iRegistry.productLookup(InEvent);
    if(allWorkers_.size()>0) {
      auto const& processName = allWorkers_[0]->description().processName();
      auto runModuleToIndicies = runLookup->indiciesForModulesInProcess(processName);
      auto lumiModuleToIndicies = lumiLookup->indiciesForModulesInProcess(processName);
      auto eventModuleToIndicies = eventLookup->indiciesForModulesInProcess(processName);
      for(auto& worker : allWorkers_) {
        worker->updateLookup(InRun,*runLookup);
        worker->updateLookup(InLumi,*lumiLookup);
        worker->updateLookup(InEvent,*eventLookup);
        worker->resolvePutIndicies(InRun,runModuleToIndicies);
        worker->resolvePutIndicies(InLumi,lumiModuleToIndicies);
        worker->resolvePutIndicies(InEvent,eventModuleToIndicies);
      }
      
      for_all(allWorkers_, std::bind(&Worker::beginJob, std::placeholders::_1));
    }
  }

  void
  WorkerManager::beginStream(StreamID iID, StreamContext& streamContext) {
    for(auto& worker: allWorkers_) {
      worker->beginStream(iID, streamContext);
    }
  }

  void
  WorkerManager::endStream(StreamID iID, StreamContext& streamContext) {
    for(auto& worker: allWorkers_) {
      worker->endStream(iID, streamContext);
    }
  }

  void
  WorkerManager::resetAll() {
    for_all(allWorkers_, std::bind(&Worker::reset, std::placeholders::_1));
  }

  void
  WorkerManager::addToAllWorkers(Worker* w) {
    if(!search_all(allWorkers_, w)) {
      allWorkers_.push_back(w);
    }
  }

  void
  WorkerManager::setupOnDemandSystem(EventPrincipal& ep, EventSetup const& es) {
    this->resetAll();
    unscheduled_.setEventSetup(es);
    if(&ep != lastSetupEventPrincipal_) {
      UnscheduledConfigurator config( allWorkers_.begin(), allWorkers_.end(), &(unscheduled_.auxiliary()));
      ep.setupUnscheduled(config);
      lastSetupEventPrincipal_ = &ep;
    }
  }
  
}
