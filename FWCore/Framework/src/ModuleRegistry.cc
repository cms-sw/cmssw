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
#include "FWCore/Framework/interface/ModuleRegistry.h"
#include "FWCore/Framework/src/ModuleHolderFactory.h"
#include "FWCore/Utilities/interface/SignalSentry.h"

namespace edm {
  std::shared_ptr<maker::ModuleHolder> ModuleRegistry::getModule(
      MakeModuleParams const& p,
      std::string const& moduleLabel,
      signalslot::Signal<void(ModuleDescription const&)>& iPre,
      signalslot::Signal<void(ModuleDescription const&)>& iPost) {
    auto modItr = labelToModule_.find(moduleLabel);
    if (modItr == labelToModule_.end()) {
      auto modPtr = ModuleHolderFactory::get()->makeModule(p, typeResolverMaker_, iPre, iPost);

      if (maxModuleID_ < modPtr->moduleDescription().id()) {
        maxModuleID_ = modPtr->moduleDescription().id();
      }
      // Transfer ownership of worker to the registry
      labelToModule_[moduleLabel] = modPtr;
      return modPtr;
    }
    return get_underlying_safe(modItr->second);
  }

  std::shared_ptr<maker::ModuleHolder> ModuleRegistry::getExistingModule(std::string const& moduleLabel) {
    auto modItr = labelToModule_.find(moduleLabel);
    if (modItr == labelToModule_.end()) {
      return {};
    }
    return get_underlying_safe(modItr->second);
  }

  maker::ModuleHolder* ModuleRegistry::replaceModule(std::string const& iModuleLabel,
                                                     edm::ParameterSet const& iPSet,
                                                     edm::PreallocationConfiguration const& iPrealloc) {
    auto modItr = labelToModule_.find(iModuleLabel);
    if (modItr == labelToModule_.end()) {
      return nullptr;
    }

    auto modPtr = ModuleHolderFactory::get()->makeReplacementModule(iPSet);
    modPtr->finishModuleInitialization(modItr->second->moduleDescription(), iPrealloc, nullptr);

    if (maxModuleID_ < modPtr->moduleDescription().id()) {
      maxModuleID_ = modPtr->moduleDescription().id();
    }

    // Transfer ownership of worker to the registry
    modItr->second = modPtr;
    return modItr->second.get();
  }

  void ModuleRegistry::deleteModule(std::string const& iModuleLabel,
                                    signalslot::Signal<void(ModuleDescription const&)>& iPre,
                                    signalslot::Signal<void(ModuleDescription const&)>& iPost) {
    auto modItr = labelToModule_.find(iModuleLabel);
    if (modItr == labelToModule_.end()) {
      throw cms::Exception("LogicError")
          << "Trying to delete module " << iModuleLabel
          << " but it does not exist in the ModuleRegistry. Please contact framework developers.";
    }
    // The pre signal must be given a reference to the ModuleDescription of the module before the module is deleted (the
    // signal is promised to be given the ModuleDescription in a stable location). After the module has been deleted
    // that stable location is gone, so the best we can do is to give a copy of the ModuleDescription.
    auto md = modItr->second->moduleDescription();
    auto guard = signalslot::make_sentry([&iPost, &md]() { iPost.emit(md); });
    iPre.emit(modItr->second->moduleDescription());
    labelToModule_.erase(modItr);
    guard.succeeded();
  }
}  // namespace edm
