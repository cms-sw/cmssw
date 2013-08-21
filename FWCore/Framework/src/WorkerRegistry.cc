/**
   \file
   Implementation of class WorkerRegistry

   \author Stefano ARGIRO
   \date 18 May 2005
*/
#include <memory>
#include "FWCore/Framework/src/WorkerRegistry.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/WorkerMaker.h"
#include "FWCore/Framework/src/Factory.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

namespace edm {
  
  class ModuleRegistry {
  public:
    std::shared_ptr<maker::ModuleHolder> getModule(WorkerParams const& p, std::string const& moduleLabel,
                    signalslot::Signal<void(ModuleDescription const&)>& iPre,
                    signalslot::Signal<void(ModuleDescription const&)>& iPost);
    
    maker::ModuleHolder* replaceModule(std::string const& iModuleLabel,
                                       edm::ParameterSet const& iPSet);
  private:
    std::map<std::string,std::shared_ptr<maker::ModuleHolder>> labelToModule_;
  };
  
  std::shared_ptr<maker::ModuleHolder>
  ModuleRegistry::getModule(WorkerParams const& p,
                            std::string const& moduleLabel,
                            signalslot::Signal<void(ModuleDescription const&)>& iPre,
                            signalslot::Signal<void(ModuleDescription const&)>& iPost) {
    auto modItr = labelToModule_.find(moduleLabel);
    if(modItr == labelToModule_.end()) {
      auto modPtr=
      Factory::get()->makeModule(p,iPre,iPost);
      
      // Transfer ownership of worker to the registry
      labelToModule_[moduleLabel] = modPtr;
      return labelToModule_[moduleLabel];
    }
    return (modItr->second);
  }
  
  maker::ModuleHolder*
  ModuleRegistry::replaceModule(std::string const& iModuleLabel,
                                edm::ParameterSet const& iPSet) {
    auto modItr = labelToModule_.find(iModuleLabel);
    if (modItr == labelToModule_.end()) {
      return nullptr;
    }
    
    auto modPtr=
    Factory::get()->makeReplacementModule(iPSet);
    modPtr->setModuleDescription(modItr->second->moduleDescription());
    
    // Transfer ownership of worker to the registry
    modItr->second = modPtr;
    return modItr->second.get();
  }
  
  boost::shared_ptr<ModuleRegistry> makeModuleRegistry() {
    return boost::shared_ptr<ModuleRegistry>(new ModuleRegistry());
  }
  
  maker::ModuleHolder* replaceModule(boost::shared_ptr<ModuleRegistry> iReg,
                                     std::string const& iModuleLabel,
                                     edm::ParameterSet const& iPSet) {
    return iReg->replaceModule(iModuleLabel,iPSet);
  }


  WorkerRegistry::WorkerRegistry(boost::shared_ptr<ActivityRegistry> areg) :
    modRegistry_(new ModuleRegistry),
    m_workerMap(),
    actReg_(areg)
  {
  }
  
  WorkerRegistry::WorkerRegistry(boost::shared_ptr<ActivityRegistry> areg,
                                 boost::shared_ptr<ModuleRegistry> modReg):
  modRegistry_(modReg),
  m_workerMap(),
  actReg_(areg)
  {
  }

  WorkerRegistry:: ~WorkerRegistry() {
  }

  void WorkerRegistry::clear() {
    m_workerMap.clear();
  }

  Worker* WorkerRegistry::getWorker(WorkerParams const& p, std::string const& moduleLabel) {

    WorkerMap::iterator workerIt = m_workerMap.find(moduleLabel);
  
    // if the worker is not there, make it
    if (workerIt == m_workerMap.end()){

      auto modulePtr = modRegistry_->getModule(p,moduleLabel,
                                               actReg_->preModuleConstructionSignal_,
                                               actReg_->postModuleConstructionSignal_);
      auto workerPtr=
        Factory::get()->makeWorker(p,modulePtr);
    
      workerPtr->setActivityRegistry(actReg_);

      // Transfer ownership of worker to the registry 
      m_workerMap[moduleLabel].reset(workerPtr.release());
      return m_workerMap[moduleLabel].get(); 
    } 
    return (workerIt->second.get());
  }
}
