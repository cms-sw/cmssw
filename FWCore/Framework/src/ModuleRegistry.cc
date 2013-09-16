// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     edm::ModuleRegistry
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Fri, 23 Aug 2013 16:39:58 GMT
//

// system include files

// user include files
#include "FWCore/Framework/src/ModuleRegistry.h"
#include "FWCore/Framework/src/Factory.h"


namespace edm {
  std::shared_ptr<maker::ModuleHolder>
  ModuleRegistry::getModule(MakeModuleParams const& p,
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
                                edm::ParameterSet const& iPSet,
                                edm::PreallocationConfiguration const& iPrealloc) {
    auto modItr = labelToModule_.find(iModuleLabel);
    if (modItr == labelToModule_.end()) {
      return nullptr;
    }
    
    auto modPtr=
    Factory::get()->makeReplacementModule(iPSet);
    modPtr->setModuleDescription(modItr->second->moduleDescription());
    modPtr->preallocate(iPrealloc);

    // Transfer ownership of worker to the registry
    modItr->second = modPtr;
    return modItr->second.get();
  }
}