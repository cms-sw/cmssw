#ifndef FWCore_Framework_ModuleRegistry_h
#define FWCore_Framework_ModuleRegistry_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     ModuleRegistry
//
/**\class edm::ModuleRegistry ModuleRegistry.h "FWCore/Framework/interface/ModuleRegistry.h"

 Description: Constructs and owns framework modules

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 23 Aug 2013 16:21:10 GMT
//

// system include files
#include <map>
#include <memory>
#include <string>

// user include files
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ModuleTypeResolverMaker.h"
#include "FWCore/Framework/interface/maker/ModuleHolder.h"

// forward declarations
namespace edm {
  class ParameterSet;
  struct MakeModuleParams;
  class ModuleDescription;
  class PreallocationConfiguration;
  class SignallingProductRegistryFiller;

  class ModuleRegistry {
  public:
    ModuleRegistry() = default;
    explicit ModuleRegistry(ModuleTypeResolverMaker const* resolverMaker) : typeResolverMaker_(resolverMaker) {}
    std::shared_ptr<maker::ModuleHolder> getModule(MakeModuleParams const& p,
                                                   std::string const& moduleLabel,
                                                   signalslot::Signal<void(ModuleDescription const&)>& iPre,
                                                   signalslot::Signal<void(ModuleDescription const&)>& iPost);
    //returns a null if module not found
    std::shared_ptr<maker::ModuleHolder> getExistingModule(std::string const& moduleLabel);

    maker::ModuleHolder* replaceModule(std::string const& iModuleLabel,
                                       edm::ParameterSet const& iPSet,
                                       edm::PreallocationConfiguration const&);

    void deleteModule(std::string const& iModuleLabel,
                      signalslot::Signal<void(ModuleDescription const&)>& iPre,
                      signalslot::Signal<void(ModuleDescription const&)>& iPost);

    template <typename T, typename... Args>
    std::shared_ptr<T> makeExplicitModule(ModuleDescription const& md,
                                          PreallocationConfiguration const& iPrealloc,
                                          SignallingProductRegistryFiller* iReg,
                                          signalslot::Signal<void(ModuleDescription const&)>& iPre,
                                          signalslot::Signal<void(ModuleDescription const&)>& iPost,
                                          Args&&... args) {
      bool postCalled = false;
      if (labelToModule_.find(md.moduleLabel()) != labelToModule_.end()) {
        throw cms::Exception("InsertError") << "Module with label '" << md.moduleLabel() << "' already exists.";
      }

      try {
        std::shared_ptr<T> module;
        convertException::wrap([&]() {
          iPre.emit(md);
          module = std::make_shared<T>(std::forward<Args>(args)...);

          auto holder = std::make_shared<maker::ModuleHolderT<typename T::ModuleType>>(module);
          holder->finishModuleInitialization(md, iPrealloc, iReg);
          labelToModule_.emplace(md.moduleLabel(), std::move(holder));

          if (maxModuleID_ < module->moduleDescription().id()) {
            maxModuleID_ = module->moduleDescription().id();
          }
          // if exception then post will be called in the catch block
          postCalled = true;
          iPost.emit(md);
        });
        return module;
      } catch (cms::Exception& iException) {
        if (!postCalled) {
          CMS_SA_ALLOW try { iPost.emit(md); } catch (...) {
            // If post throws an exception ignore it because we are already handling another exception
          }
        }
        throw;
      }
    }

    template <typename F>
    void forAllModuleHolders(F iFunc) {
      for (auto& labelMod : labelToModule_) {
        maker::ModuleHolder* t = labelMod.second.get();
        iFunc(t);
      }
    }

    template <typename F>
    void forAllModuleHolders(F iFunc) const {
      for (auto& labelMod : labelToModule_) {
        maker::ModuleHolder const* t = labelMod.second.get();
        iFunc(t);
      }
    }

    unsigned int maxModuleID() const { return maxModuleID_; }

  private:
    std::map<std::string, edm::propagate_const<std::shared_ptr<maker::ModuleHolder>>> labelToModule_;
    ModuleTypeResolverMaker const* typeResolverMaker_;
    unsigned int maxModuleID_ = 0;
  };
}  // namespace edm

#endif
