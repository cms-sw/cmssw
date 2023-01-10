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
#include "FWCore/Framework/interface/maker/MakeModuleParams.h"
#include "FWCore/Framework/interface/ModuleRegistry.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

namespace edm {

  WorkerRegistry::WorkerRegistry(std::shared_ptr<ActivityRegistry> areg, ModuleTypeResolverMaker const* resolverMaker)
      : modRegistry_(std::make_shared<ModuleRegistry>(resolverMaker)), m_workerMap(), actReg_(areg) {}

  WorkerRegistry::WorkerRegistry(std::shared_ptr<ActivityRegistry> areg, std::shared_ptr<ModuleRegistry> modReg)
      : modRegistry_(modReg), m_workerMap(), actReg_(areg) {}

  WorkerRegistry::~WorkerRegistry() {}

  void WorkerRegistry::clear() { m_workerMap.clear(); }

  Worker* WorkerRegistry::getWorker(WorkerParams const& p, std::string const& moduleLabel) {
    WorkerMap::iterator workerIt = m_workerMap.find(moduleLabel);

    // if the worker is not there, make it
    if (workerIt == m_workerMap.end()) {
      MakeModuleParams mmp(p.pset_, *p.reg_, p.preallocate_, p.processConfiguration_);
      auto modulePtr = modRegistry_->getModule(
          mmp, moduleLabel, actReg_->preModuleConstructionSignal_, actReg_->postModuleConstructionSignal_);
      auto workerPtr = modulePtr->makeWorker(p.actions_);

      workerPtr->setActivityRegistry(actReg_);

      // Transfer ownership of worker to the registry
      m_workerMap[moduleLabel] =
          std::shared_ptr<Worker>(workerPtr.release());  // propagate_const<T> has no reset() function
      return m_workerMap[moduleLabel].get();
    }
    return (workerIt->second.get());
  }

  Worker const* WorkerRegistry::get(std::string const& moduleLabel) const {
    WorkerMap::const_iterator workerIt = m_workerMap.find(moduleLabel);
    if (workerIt != m_workerMap.end()) {
      return workerIt->second;
    }
    return nullptr;
  }

  void WorkerRegistry::deleteModule(std::string const& moduleLabel) {
    WorkerMap::iterator workerIt = m_workerMap.find(moduleLabel);
    if (workerIt == m_workerMap.end()) {
      throw cms::Exception("LogicError")
          << "WorkerRegistry::deleteModule() Trying to delete the module of a Worker with label " << moduleLabel
          << " but no such Worker exists in the WorkerRegistry. Please contact framework developers.";
    }
    workerIt->second->clearModule();
  }

}  // namespace edm
