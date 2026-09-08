#include "FWCore/Framework/interface/WorkerManager.h"
#include "FWCore/Framework/interface/WorkerManager_stream.h"
#include "FWCore/Framework/interface/WorkerManager_global.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/Framework/interface/TransitionPhaseTypes.h"
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
  template <typename TI, typename TP>
  WorkerManagerCore<TI, TP>::WorkerManagerCore(std::shared_ptr<ModuleRegistry> modReg,
                                               std::shared_ptr<ActivityRegistry> areg,
                                               ExceptionToActionTable const& actions)
      : workerReg_(areg, modReg), actionTable_(&actions), allWorkers_(), lastSetupPrincipal_(nullptr) {}

  template <typename TI>
  WorkerManager<TI, TransitionPhaseGlobal>::WorkerManager(std::shared_ptr<ModuleRegistry> modReg,
                                                          std::shared_ptr<ActivityRegistry> areg,
                                                          ExceptionToActionTable const& actions)
      : WorkerManagerCore<TI, TransitionPhaseGlobal>(modReg, areg, actions) {}

  template <typename TI>
  WorkerManager<TI, TransitionPhaseStream>::WorkerManager(std::shared_ptr<ModuleRegistry> modReg,
                                                          std::shared_ptr<ActivityRegistry> areg,
                                                          ExceptionToActionTable const& actions)
      : WorkerManagerCore<TI, TransitionPhaseStream>(modReg, areg, actions) {}

  WorkerManager<EventTransitionInfo, TransitionPhaseGlobal>::WorkerManager(std::shared_ptr<ModuleRegistry> modReg,
                                                                           std::shared_ptr<ActivityRegistry> areg,
                                                                           ExceptionToActionTable const& actions)
      : WorkerManagerCore<EventTransitionInfo, TransitionPhaseGlobal>(modReg, areg, actions),
        unscheduled_(*areg) {}  // WorkerManager::WorkerManager

  template <typename TI, typename TP>
  Worker const* WorkerManagerCore<TI, TP>::deleteModuleIfExists(std::string const& moduleLabel) {
    auto worker = workerReg_.get(moduleLabel);
    if (worker != nullptr) {
      auto eraseBeg = std::remove(allWorkers_.begin(), allWorkers_.end(), worker);
      allWorkers_.erase(eraseBeg, allWorkers_.end());
      workerReg_.deleteModule(moduleLabel);
    }
    return worker;
  }

  template <typename TI>
  void WorkerManager<TI, TransitionPhaseStream>::deleteModuleIfExists(std::string const& moduleLabel) {
    (void)WorkerManagerCore<TI, TransitionPhaseStream>::deleteModuleIfExists(moduleLabel);
  }
  template <typename TI>
  void WorkerManager<TI, TransitionPhaseGlobal>::deleteModuleIfExists(std::string const& moduleLabel) {
    (void)WorkerManagerCore<TI, TransitionPhaseGlobal>::deleteModuleIfExists(moduleLabel);
  }

  void WorkerManager<EventTransitionInfo, TransitionPhaseGlobal>::deleteModuleIfExists(std::string const& moduleLabel) {
    auto worker = WorkerManagerCore<EventTransitionInfo, TransitionPhaseGlobal>::deleteModuleIfExists(moduleLabel);
    if (worker != nullptr) {
      unscheduled_.removeWorker(worker);
    }
  }

  template <typename TI, typename TP>
  Worker* WorkerManagerCore<TI, TP>::getWorkerForExistingModule(std::string const& label) {
    auto worker = workerReg_.getWorkerFromExistingModule(label, actionTable_);
    if (nullptr != worker) {
      addToAllWorkers(worker);
    }
    return worker;
  }

  void WorkerManager<EventTransitionInfo, TransitionPhaseGlobal>::addToUnscheduledWorkers(
      ModuleDescription const& iDescription) {
    auto newWorker = this->getWorkerForExistingModuleUnattached(iDescription.moduleLabel());
    assert(nullptr != newWorker);
    assert(newWorker->moduleType() == Worker::Types::kProducer || newWorker->moduleType() == Worker::Types::kFilter);
    unscheduled_.addWorker(newWorker);
    //add to list so it gets reset each new event
    addToAllWorkers(newWorker);
  }

  template <typename TI, typename TP>
  void WorkerManagerCore<TI, TP>::resetAll() {
    for_all(allWorkers_, std::bind(&Worker::reset, std::placeholders::_1));
  }

  template <typename TI, typename TP>
  void WorkerManagerCore<TI, TP>::addToAllWorkers(Worker* w) {
    if (!search_all(allWorkers_, w)) {
      allWorkers_.push_back(w);
    }
  }

  template <typename TI, typename TP>
  void WorkerManagerCore<TI, TP>::setupResolvers(Principal& ep, UnscheduledAuxiliary const* aux) {
    if (&ep != lastSetupPrincipal_) {
      UnscheduledConfigurator config(allWorkers().begin(), allWorkers().end(), aux);
      ep.setupUnscheduled(config);
      lastSetupPrincipal_ = &ep;
    }
  }

  template <typename TI>
  void WorkerManager<TI, TransitionPhaseStream>::setupResolvers(Principal& ep) {
    this->setupResolvers(ep, nullptr);
  }
  template <typename TI>
  void WorkerManager<TI, TransitionPhaseGlobal>::setupResolvers(Principal& ep) {
    this->setupResolvers(ep, nullptr);
  }

  void WorkerManager<EventTransitionInfo, TransitionPhaseGlobal>::setupResolvers(Principal& ep) {
    this->setupResolvers(ep, &(unscheduled_.auxiliary()));
  }

  void WorkerManager<EventTransitionInfo, TransitionPhaseGlobal>::setupOnDemandSystem(EventTransitionInfo const& info) {
    unscheduled_.setEventTransitionInfo(info);
  }

  template class WorkerManagerCore<RunTransitionInfo, TransitionPhaseGlobal>;
  template class WorkerManagerCore<LumiTransitionInfo, TransitionPhaseGlobal>;
  template class WorkerManagerCore<RunTransitionInfo, TransitionPhaseStream>;
  template class WorkerManagerCore<LumiTransitionInfo, TransitionPhaseStream>;
  template class WorkerManagerCore<EventTransitionInfo, TransitionPhaseGlobal>;
  template class WorkerManagerCore<ProcessBlockTransitionInfo, TransitionPhaseGlobal>;
  template class WorkerManagerCore<InputProcessBlockTransitionInfo, TransitionPhaseGlobal>;

  template class WorkerManager<RunTransitionInfo, TransitionPhaseGlobal>;
  template class WorkerManager<LumiTransitionInfo, TransitionPhaseGlobal>;
  template class WorkerManager<RunTransitionInfo, TransitionPhaseStream>;
  template class WorkerManager<LumiTransitionInfo, TransitionPhaseStream>;
  template class WorkerManager<EventTransitionInfo, TransitionPhaseGlobal>;
  template class WorkerManager<ProcessBlockTransitionInfo, TransitionPhaseGlobal>;
  template class WorkerManager<InputProcessBlockTransitionInfo, TransitionPhaseGlobal>;
}  // namespace edm
