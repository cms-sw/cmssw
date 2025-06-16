#ifndef FWCore_Framework_ModuleHolder_h
#define FWCore_Framework_ModuleHolder_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     ModuleHolder
//
/**\class edm::maker::ModuleHolder ModuleHolder.h "FWCore/Framework/interface/maker/ModuleHolder.h"

 Description: Base class used to own a module for the framework

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 23 Aug 2013 17:47:04 GMT
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/maker/WorkerT.h"
#include "FWCore/Framework/interface/OutputModuleCommunicatorT.h"
#include "FWCore/Framework/interface/SignallingProductRegistryFiller.h"
// forward declarations
namespace edm {
  class ModuleDescription;
  class SignallingProductRegistryFiller;
  class ExceptionToActionTable;
  class PreallocationConfiguration;

  namespace maker {
    class ModuleHolder {
    public:
      ModuleHolder() = default;
      virtual ~ModuleHolder() {}
      virtual std::unique_ptr<Worker> makeWorker(ExceptionToActionTable const* actions) const = 0;

      virtual ModuleDescription const& moduleDescription() const = 0;
      virtual void finishModuleInitialization(ModuleDescription const& iDesc,
                                              PreallocationConfiguration const& iPrealloc,
                                              SignallingProductRegistryFiller* iReg) = 0;
      virtual void replaceModuleFor(Worker*) const = 0;

      virtual std::unique_ptr<OutputModuleCommunicator> createOutputModuleCommunicator() = 0;
    };

    template <typename T>
    class ModuleHolderT : public ModuleHolder {
    public:
      ModuleHolderT(std::shared_ptr<T> iModule) : m_mod(iModule) {}
      ~ModuleHolderT() override {}
      std::shared_ptr<T> module() const { return m_mod; }
      void replaceModuleFor(Worker* iWorker) const override {
        auto w = dynamic_cast<WorkerT<T>*>(iWorker);
        assert(nullptr != w);
        w->setModule(m_mod);
      }
      std::unique_ptr<Worker> makeWorker(ExceptionToActionTable const* actions) const override {
        return std::make_unique<edm::WorkerT<T>>(module(), moduleDescription(), actions);
      }

      static void finishModuleInitialization(T& iModule,
                                             ModuleDescription const& iDesc,
                                             PreallocationConfiguration const& iPrealloc,
                                             SignallingProductRegistryFiller* iReg) {
        iModule.setModuleDescription(iDesc);
        iModule.doPreallocate(iPrealloc);
        if (iReg) {
          iModule.registerProductsAndCallbacks(&iModule, iReg);
        }
      };
      ModuleDescription const& moduleDescription() const override { return m_mod->moduleDescription(); }

      void finishModuleInitialization(ModuleDescription const& iDesc,
                                      PreallocationConfiguration const& iPrealloc,
                                      SignallingProductRegistryFiller* iReg) override {
        finishModuleInitialization(*m_mod, iDesc, iPrealloc, iReg);
      }
      std::unique_ptr<OutputModuleCommunicator> createOutputModuleCommunicator() override {
        return OutputModuleCommunicatorT<T>::createIfNeeded(m_mod.get());
      }

    private:
      std::shared_ptr<T> m_mod;
    };
  }  // namespace maker
}  // namespace edm

#endif
