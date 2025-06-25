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
#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"

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
    // If iPost throws and exception, let it propagate
    // If deletion throws an exception, capture it and call iPost before throwing an exception
    // If iPost throws an exception, let it propagate
    auto md = modItr->second->moduleDescription();
    iPre(modItr->second->moduleDescription());
    bool postCalled = false;
    // exception is rethrown
    CMS_SA_ALLOW try {
      labelToModule_.erase(modItr);
      // if exception then post will be called in the catch block
      postCalled = true;
      iPost(md);
    } catch (...) {
      if (not postCalled) {
        // we're already handling exception, nothing we can do if iPost throws
        CMS_SA_ALLOW try { iPost(md); } catch (...) {
        }
      }
      throw;
    }
  }

  void ModuleRegistry::finishModulesInitialization(ProductRegistry const& iRegistry,
                                                   eventsetup::ESRecordsToProductResolverIndices const& iESIndices,
                                                   ProcessBlockHelperBase const& processBlockHelperBase,
                                                   std::string const& processName) {
    auto const processBlockLookup = iRegistry.productLookup(InProcess);
    auto const runLookup = iRegistry.productLookup(InRun);
    auto const lumiLookup = iRegistry.productLookup(InLumi);
    auto const eventLookup = iRegistry.productLookup(InEvent);

    auto processBlockModuleToIndicies = processBlockLookup->indiciesForModulesInProcess(processName);
    auto runModuleToIndicies = runLookup->indiciesForModulesInProcess(processName);
    auto lumiModuleToIndicies = lumiLookup->indiciesForModulesInProcess(processName);
    auto eventModuleToIndicies = eventLookup->indiciesForModulesInProcess(processName);

    forAllModuleHolders([&](auto& iHolder) {
      iHolder->updateLookup(InProcess, *processBlockLookup);
      iHolder->updateLookup(InRun, *runLookup);
      iHolder->updateLookup(InLumi, *lumiLookup);
      iHolder->updateLookup(InEvent, *eventLookup);
      iHolder->updateLookup(iESIndices);
      iHolder->resolvePutIndicies(InProcess, processBlockModuleToIndicies);
      iHolder->resolvePutIndicies(InRun, runModuleToIndicies);
      iHolder->resolvePutIndicies(InLumi, lumiModuleToIndicies);
      iHolder->resolvePutIndicies(InEvent, eventModuleToIndicies);
      iHolder->selectInputProcessBlocks(iRegistry, processBlockHelperBase);
    });
  }
}  // namespace edm
