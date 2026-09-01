/**
   \file
   Implementation of class WorkerRegistry

   \author Stefano ARGIRO
   \date 18 May 2005
*/
#include <memory>
#include "FWCore/Framework/interface/WorkerRegistry.h"
#include "FWCore/Framework/interface/maker/Worker.h"
#include "FWCore/Framework/interface/maker/ModuleHolder.h"
#include "FWCore/Framework/interface/ModuleRegistry.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/Framework/interface/TransitionPhaseTypes.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

namespace edm {

  template <typename TI, typename TP>
  WorkerRegistry<TI, TP>::WorkerRegistry(std::shared_ptr<ActivityRegistry> areg, std::shared_ptr<ModuleRegistry> modReg)
      : modRegistry_(modReg), m_workerMap(), actReg_(areg) {}

  template <typename TI, typename TP>
  WorkerRegistry<TI, TP>::~WorkerRegistry() {}

  template <typename TI, typename TP>
  void WorkerRegistry<TI, TP>::clear() {
    m_workerMap.clear();
  }

  template <typename TI, typename TP>
  Worker const* WorkerRegistry<TI, TP>::get(std::string const& moduleLabel) const {
    WorkerMap::const_iterator workerIt = m_workerMap.find(moduleLabel);
    if (workerIt != m_workerMap.end()) {
      return workerIt->second;
    }
    return nullptr;
  }

  template <typename TI, typename TP>
  Worker* WorkerRegistry<TI, TP>::getWorkerFromExistingModule(std::string const& moduleLabel,
                                                              ExceptionToActionTable const* actions) {
    WorkerMap::iterator workerIt = m_workerMap.find(moduleLabel);
    if (workerIt == m_workerMap.end()) {
      auto modulePtr = modRegistry_->getExistingModule(moduleLabel);
      if (!modulePtr) {
        return nullptr;
      }
      auto workerPtr = modulePtr->makeWorker(actions, TI::key(), TP::value);

      workerPtr->setActivityRegistry(actReg_);

      // Transfer ownership of worker to the registry
      m_workerMap[moduleLabel] =
          std::shared_ptr<Worker>(workerPtr.release());  // propagate_const<T> has no reset() function
      return m_workerMap[moduleLabel].get();
    }
    return (workerIt->second.get());
  }
  template <typename TI, typename TP>
  void WorkerRegistry<TI, TP>::deleteModule(std::string const& moduleLabel) {
    WorkerMap::iterator workerIt = m_workerMap.find(moduleLabel);
    if (workerIt == m_workerMap.end()) {
      throw cms::Exception("LogicError")
          << "WorkerRegistry::deleteModule() Trying to delete the module of a Worker with label " << moduleLabel
          << " but no such Worker exists in the WorkerRegistry. Please contact framework developers.";
    }
    workerIt->second->clearModule();
  }

  template class WorkerRegistry<edm::RunTransitionInfo, edm::TransitionPhaseGlobal>;
  template class WorkerRegistry<edm::LumiTransitionInfo, edm::TransitionPhaseGlobal>;
  template class WorkerRegistry<edm::ProcessBlockTransitionInfo, edm::TransitionPhaseGlobal>;
  template class WorkerRegistry<edm::InputProcessBlockTransitionInfo, edm::TransitionPhaseGlobal>;
  template class WorkerRegistry<edm::RunTransitionInfo, edm::TransitionPhaseStream>;
  template class WorkerRegistry<edm::LumiTransitionInfo, edm::TransitionPhaseStream>;
  template class WorkerRegistry<edm::EventTransitionInfo, edm::TransitionPhaseStream>;
}  // namespace edm
