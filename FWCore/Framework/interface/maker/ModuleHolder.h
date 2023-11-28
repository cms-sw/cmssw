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

// forward declarations
namespace edm {
  class Maker;
  class ModuleDescription;
  class ProductRegistry;
  class ExceptionToActionTable;
  class PreallocationConfiguration;

  namespace maker {
    class ModuleHolder {
    public:
      ModuleHolder(Maker const* iMaker) : m_maker(iMaker) {}
      virtual ~ModuleHolder() {}
      std::unique_ptr<Worker> makeWorker(ExceptionToActionTable const* actions) const;

      virtual ModuleDescription const& moduleDescription() const = 0;
      virtual void setModuleDescription(ModuleDescription const& iDesc) = 0;
      virtual void preallocate(PreallocationConfiguration const&) = 0;
      virtual void registerProductsAndCallbacks(ProductRegistry*) = 0;
      virtual void replaceModuleFor(Worker*) const = 0;

      virtual std::unique_ptr<OutputModuleCommunicator> createOutputModuleCommunicator() = 0;

    protected:
      Maker const* m_maker;
    };

    template <typename T>
    class ModuleHolderT : public ModuleHolder {
    public:
      ModuleHolderT(std::shared_ptr<T> iModule, Maker const* iMaker) : ModuleHolder(iMaker), m_mod(iModule) {}
      ~ModuleHolderT() override {}
      std::shared_ptr<T> module() const { return m_mod; }
      void replaceModuleFor(Worker* iWorker) const override {
        auto w = dynamic_cast<WorkerT<T>*>(iWorker);
        assert(nullptr != w);
        w->setModule(m_mod);
      }
      ModuleDescription const& moduleDescription() const override { return m_mod->moduleDescription(); }
      void setModuleDescription(ModuleDescription const& iDesc) override { m_mod->setModuleDescription(iDesc); }
      void preallocate(PreallocationConfiguration const& iPrealloc) override { m_mod->doPreallocate(iPrealloc); }

      void registerProductsAndCallbacks(ProductRegistry* iReg) override {
        m_mod->registerProductsAndCallbacks(module().get(), iReg);
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
