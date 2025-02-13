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

  WorkerManager::WorkerManager(std::shared_ptr<ActivityRegistry> areg,
                               ExceptionToActionTable const& actions,
                               ModuleTypeResolverMaker const* typeResolverMaker)
      : workerReg_(areg, typeResolverMaker),
        actionTable_(&actions),
        allWorkers_(),
        unscheduled_(*areg),
        lastSetupEventPrincipal_(nullptr) {}  // WorkerManager::WorkerManager

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
                                   SignallingProductRegistry& preg,
                                   PreallocationConfiguration const* prealloc,
                                   std::shared_ptr<ProcessConfiguration const> processConfiguration,
                                   std::string const& label) {
    WorkerParams params(&pset, preg, prealloc, processConfiguration, *actionTable_);
    return workerReg_.getWorker(params, label);
  }

  void WorkerManager::addToUnscheduledWorkers(ParameterSet& pset,
                                              SignallingProductRegistry& preg,
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

  void WorkerManager::releaseMemoryPostLookupSignal() {
    for (auto& worker : allWorkers_) {
      worker->releaseMemoryPostLookupSignal();
    }
  }

  void WorkerManager::beginJob(ProductRegistry const& iRegistry,
                               eventsetup::ESRecordsToProductResolverIndices const& iESIndices,
                               ProcessBlockHelperBase const& processBlockHelperBase,
                               GlobalContext const& globalContext) {
    std::exception_ptr exceptionPtr;
    CMS_SA_ALLOW try {
      auto const processBlockLookup = iRegistry.productLookup(InProcess);
      auto const runLookup = iRegistry.productLookup(InRun);
      auto const lumiLookup = iRegistry.productLookup(InLumi);
      auto const eventLookup = iRegistry.productLookup(InEvent);
      if (!allWorkers_.empty()) {
        auto const& processName = allWorkers_[0]->description()->processName();
        auto processBlockModuleToIndicies = processBlockLookup->indiciesForModulesInProcess(processName);
        auto runModuleToIndicies = runLookup->indiciesForModulesInProcess(processName);
        auto lumiModuleToIndicies = lumiLookup->indiciesForModulesInProcess(processName);
        auto eventModuleToIndicies = eventLookup->indiciesForModulesInProcess(processName);
        for (auto& worker : allWorkers_) {
          worker->updateLookup(InProcess, *processBlockLookup);
          worker->updateLookup(InRun, *runLookup);
          worker->updateLookup(InLumi, *lumiLookup);
          worker->updateLookup(InEvent, *eventLookup);
          worker->updateLookup(iESIndices);
          worker->resolvePutIndicies(InProcess, processBlockModuleToIndicies);
          worker->resolvePutIndicies(InRun, runModuleToIndicies);
          worker->resolvePutIndicies(InLumi, lumiModuleToIndicies);
          worker->resolvePutIndicies(InEvent, eventModuleToIndicies);
          worker->selectInputProcessBlocks(iRegistry, processBlockHelperBase);
        }
      }
    } catch (...) {
      exceptionPtr = std::current_exception();
    }

    for (auto& worker : allWorkers_) {
      CMS_SA_ALLOW try { worker->beginJob(globalContext); } catch (...) {
        if (!exceptionPtr) {
          exceptionPtr = std::current_exception();
        }
      }
    }
    if (exceptionPtr) {
      std::rethrow_exception(exceptionPtr);
    }
  }

  void WorkerManager::endJob(ExceptionCollector& collector, GlobalContext const& globalContext) {
    for (auto& worker : allWorkers_) {
      try {
        convertException::wrap([&worker, &globalContext]() { worker->endJob(globalContext); });
      } catch (cms::Exception const& ex) {
        collector.addException(ex);
      }
    }
  }

  void WorkerManager::beginStream(StreamID streamID, StreamContext const& streamContext) {
    std::exception_ptr exceptionPtr;
    for (auto& worker : allWorkers_) {
      CMS_SA_ALLOW try { worker->beginStream(streamID, streamContext); } catch (...) {
        if (!exceptionPtr) {
          exceptionPtr = std::current_exception();
        }
      }
    }
    if (exceptionPtr) {
      std::rethrow_exception(exceptionPtr);
    }
  }

  void WorkerManager::endStream(StreamID streamID,
                                StreamContext const& streamContext,
                                ExceptionCollector& collector,
                                std::mutex& collectorMutex) noexcept {
    for (auto& worker : allWorkers_) {
      CMS_SA_ALLOW try { worker->endStream(streamID, streamContext); } catch (...) {
        std::exception_ptr exceptionPtr = std::current_exception();
        std::lock_guard<std::mutex> collectorLock(collectorMutex);
        collector.call([&exceptionPtr]() { std::rethrow_exception(exceptionPtr); });
      }
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
