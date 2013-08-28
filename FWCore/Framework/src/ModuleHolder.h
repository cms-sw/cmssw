#ifndef FWCore_Framework_ModuleHolder_h
#define FWCore_Framework_ModuleHolder_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     ModuleHolder
// 
/**\class edm::maker::ModuleHolder ModuleHolder.h "FWCore/Framework/src/ModuleHolder.h"

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
#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/Framework/src/OutputModuleCommunicatorT.h"

// forward declarations
namespace edm {
  class Maker;
  class ModuleDescription;
  class ProductRegistry;
  class ExceptionToActionTable;

  namespace maker {
    class ModuleHolder {
    public:
      ModuleHolder(void* iModule, Maker const* iMaker): m_mod(iModule),
      m_maker(iMaker){}
      virtual ~ModuleHolder() {}
      std::unique_ptr<Worker> makeWorker(ExceptionToActionTable const* actions) const;
      
      virtual ModuleDescription const& moduleDescription() const = 0;
      virtual void setModuleDescription(ModuleDescription const& iDesc) = 0;
      virtual void registerProductsAndCallbacks(ProductRegistry*)=0;
      virtual void replaceModuleFor(Worker*) const = 0;

      virtual std::unique_ptr<OutputModuleCommunicator> createOutputModuleCommunicator() = 0;
    protected:
      void * m_mod;
      Maker const* m_maker;
    };
    
    template<typename T>
    class ModuleHolderT : public ModuleHolder {
    public:
      ModuleHolderT(T* iModule, Maker const* iMaker) :ModuleHolder(iModule,iMaker) {}
      ~ModuleHolderT() { delete reinterpret_cast<T*>(m_mod); }
      T* module() const { return reinterpret_cast<T*>(m_mod); }
      void replaceModuleFor(Worker* iWorker) const override {
        auto w = dynamic_cast<WorkerT<T>*>(iWorker);
        assert(0!=w);
        w->setModule(module());
      }
      ModuleDescription const& moduleDescription() const override {
        return module()->moduleDescription();
      }
      void setModuleDescription(ModuleDescription const& iDesc) override {
        module()->setModuleDescription(iDesc);
      }
      void registerProductsAndCallbacks(ProductRegistry* iReg) override {
        module()->registerProductsAndCallbacks(module(),iReg);
      }
      T* release() {
        T* m = module();
        m_mod = nullptr;
        return m;
      }
      
      std::unique_ptr<OutputModuleCommunicator>
      createOutputModuleCommunicator() {
        return std::move(OutputModuleCommunicatorT<T>::createIfNeeded(this->module()));
      }

    };
  }
}

#endif
