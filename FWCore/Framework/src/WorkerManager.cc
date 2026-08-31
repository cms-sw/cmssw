#include "FWCore/Framework/interface/WorkerManager.h"
#include "UnscheduledConfigurator.h"

#include "FWCore/Framework/interface/maker/Worker.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <exception>
#include <functional>

namespace edm {
  // -----------------------------
  template <typename TI>
  WorkerManager<TI>::WorkerManager(std::shared_ptr<ModuleRegistry> modReg,
                                   std::shared_ptr<ActivityRegistry> areg,
                                   ExceptionToActionTable const& actions)
      : workerReg_(areg, modReg),
        actionTable_(&actions),
        allWorkers_(),
        unscheduled_(*areg),
        lastSetupEventPrincipal_(nullptr) {}  // WorkerManager::WorkerManager

  template <typename TI>
  void WorkerManager<TI>::deleteModuleIfExists(std::string const& moduleLabel) {
    auto worker = workerReg_.get(moduleLabel);
    if (worker != nullptr) {
      auto eraseBeg = std::remove(allWorkers_.begin(), allWorkers_.end(), worker);
      allWorkers_.erase(eraseBeg, allWorkers_.end());
      unscheduled_.removeWorker(worker);
      workerReg_.deleteModule(moduleLabel);
    }
  }

  template <typename TI>
  Worker* WorkerManager<TI>::getWorkerForExistingModule(std::string const& label) {
    auto worker = workerReg_.getWorkerFromExistingModule(label, actionTable_);
    if (nullptr != worker) {
      addToAllWorkers(worker);
    }
    return worker;
  }

  template <typename TI>
  void WorkerManager<TI>::addToUnscheduledWorkers(ModuleDescription const& iDescription) {
    auto newWorker = workerReg_.getWorkerFromExistingModule(iDescription.moduleLabel(), actionTable_);
    assert(nullptr != newWorker);
    assert(newWorker->moduleType() == Worker::kProducer || newWorker->moduleType() == Worker::kFilter);
    unscheduled_.addWorker(newWorker);
    //add to list so it gets reset each new event
    addToAllWorkers(newWorker);
  }

  template <typename TI>
  void WorkerManager<TI>::resetAll() {
    for_all(allWorkers_, std::bind(&Worker::reset, std::placeholders::_1));
  }

  template <typename TI>
  void WorkerManager<TI>::addToAllWorkers(Worker* w) {
    if (!search_all(allWorkers_, w)) {
      allWorkers_.push_back(w);
    }
  }

  template <typename TI>
  void WorkerManager<TI>::setupResolvers(Principal& ep) {
    this->resetAll();
    if (&ep != lastSetupEventPrincipal_) {
      UnscheduledConfigurator config(allWorkers_.begin(), allWorkers_.end(), &(unscheduled_.auxiliary()));
      ep.setupUnscheduled(config);
      lastSetupEventPrincipal_ = &ep;
    }
  }

  template <typename TI>
  void WorkerManager<TI>::setupOnDemandSystem(EventTransitionInfo const& info) {
    unscheduled_.setEventTransitionInfo(info);
  }

  template class WorkerManager<RunTransitionInfo>;
  template class WorkerManager<LumiTransitionInfo>;
  template class WorkerManager<EventTransitionInfo>;
  template class WorkerManager<ProcessBlockTransitionInfo>;
}  // namespace edm
