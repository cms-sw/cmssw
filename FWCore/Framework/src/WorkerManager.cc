#include "FWCore/Framework/interface/WorkerManager.h"
#include "UnscheduledConfigurator.h"

#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"
#include "FWCore/Framework/interface/maker/Worker.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ExceptionCollector.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <exception>
#include <functional>

static const std::string kFilterType("EDFilter");
static const std::string kProducerType("EDProducer");

namespace edm {
  // -----------------------------
  WorkerManager::WorkerManager(std::shared_ptr<ModuleRegistry> modReg,
                               std::shared_ptr<ActivityRegistry> areg,
                               ExceptionToActionTable const& actions)
      : workerReg_(areg, modReg),
        actionTable_(&actions),
        allWorkers_(),
        unscheduled_(*areg),
        lastSetupEventPrincipal_(nullptr) {}  // WorkerManager::WorkerManager

  void WorkerManager::deleteModuleIfExists(std::string const& moduleLabel) {
    auto worker = workerReg_.get(moduleLabel);
    if (worker != nullptr) {
      auto eraseBeg = std::remove(allWorkers_.begin(), allWorkers_.end(), worker);
      allWorkers_.erase(eraseBeg, allWorkers_.end());
      unscheduled_.removeWorker(worker);
      workerReg_.deleteModule(moduleLabel);
    }
  }

  Worker* WorkerManager::getWorker(ParameterSet& pset,
                                   SignallingProductRegistryFiller& preg,
                                   PreallocationConfiguration const* prealloc,
                                   std::shared_ptr<ProcessConfiguration const> processConfiguration,
                                   std::string const& label,
                                   bool addToAll) {
    WorkerParams params(&pset, preg, prealloc, processConfiguration, *actionTable_);
    auto worker = workerReg_.getWorker(params, label);
    if (nullptr != worker and addToAll) {
      addToAllWorkers(worker);
    }
    return worker;
  }

  Worker* WorkerManager::getWorkerForExistingModule(std::string const& label) {
    auto worker = workerReg_.getWorkerFromExistingModule(label, actionTable_);
    if (nullptr != worker) {
      addToAllWorkers(worker);
    }
    return worker;
  }

  void WorkerManager::addToUnscheduledWorkers(ParameterSet& pset,
                                              SignallingProductRegistryFiller& preg,
                                              PreallocationConfiguration const* prealloc,
                                              std::shared_ptr<ProcessConfiguration const> processConfiguration,
                                              std::string label,
                                              std::set<std::string>& unscheduledLabels,
                                              std::vector<std::string>& shouldBeUsedLabels) {
    //Need to
    // 1) create worker
    // 2) if it is a WorkerT<EDProducer>, add it to our list
    auto modType = pset.getParameter<std::string>("@module_edm_type");
    if (modType == kProducerType || modType == kFilterType) {
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

  void WorkerManager::resetAll() { for_all(allWorkers_, std::bind(&Worker::reset, std::placeholders::_1)); }

  void WorkerManager::addToAllWorkers(Worker* w) {
    if (!search_all(allWorkers_, w)) {
      allWorkers_.push_back(w);
    }
  }

  void WorkerManager::setupResolvers(Principal& ep) {
    this->resetAll();
    if (&ep != lastSetupEventPrincipal_) {
      UnscheduledConfigurator config(allWorkers_.begin(), allWorkers_.end(), &(unscheduled_.auxiliary()));
      ep.setupUnscheduled(config);
      lastSetupEventPrincipal_ = &ep;
    }
  }

  void WorkerManager::setupOnDemandSystem(EventTransitionInfo const& info) {
    unscheduled_.setEventTransitionInfo(info);
  }

}  // namespace edm
