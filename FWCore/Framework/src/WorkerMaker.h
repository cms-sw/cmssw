#ifndef FWCore_Framework_WorkerMaker_h
#define FWCore_Framework_WorkerMaker_h

#include <cassert>
#include <memory>
#include <string>

#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/Framework/src/WorkerParams.h"

#include "FWCore/Utilities/interface/Signal.h"


namespace edm {
  class ConfigurationDescriptions;
  class ModuleDescription;
  class ParameterSet;
  class ProductRegistry;
  
  namespace maker {
    class ModuleHolder {
    public:
      explicit ModuleHolder(void* iModule): m_mod(iModule) {}
      virtual ~ModuleHolder() {}
      virtual ModuleDescription const& moduleDescription() const = 0;
      virtual void setModuleDescription(ModuleDescription const& iDesc) = 0;
      virtual void registerProductsAndCallbacks(ProductRegistry*)=0;
      virtual void replaceModuleFor(Worker*) const = 0;
    protected:
      void * m_mod;
    };
    
    template<typename T>
    class ModuleHolderT : public ModuleHolder {
    public:
      explicit ModuleHolderT( T* iModule):ModuleHolder(iModule) {}
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
    };
  }
  
  class Maker {
  public:
    virtual ~Maker();
    std::shared_ptr<maker::ModuleHolder> makeModule(WorkerParams const&,
                                       signalslot::Signal<void(ModuleDescription const&)>& iPre,
                                       signalslot::Signal<void(ModuleDescription const&)>& iPost) const;
    std::unique_ptr<Worker> makeWorker(WorkerParams const&,
                                       std::shared_ptr<maker::ModuleHolder>) const;

    std::shared_ptr<maker::ModuleHolder> makeReplacementModule(edm::ParameterSet const& p) const { return makeModule(p);}
protected:
    ModuleDescription createModuleDescription(WorkerParams const& p) const;

    void throwConfigurationException(ModuleDescription const& md,
                                     signalslot::Signal<void(ModuleDescription const&)>& post,
                                     cms::Exception & iException) const;

    void throwValidationException(WorkerParams const& p,
				  cms::Exception & iException) const;

    void validateEDMType(std::string const& edmType, WorkerParams const& p) const;

  private:
    virtual void fillDescriptions(ConfigurationDescriptions& iDesc) const = 0;
    virtual std::shared_ptr<maker::ModuleHolder> makeModule(edm::ParameterSet const& p) const  = 0;
    virtual std::unique_ptr<Worker> makeWorker(WorkerParams const& p,
                                             ModuleDescription const& md,
                                               std::shared_ptr<maker::ModuleHolder> mod) const = 0;
    virtual const std::string& baseType() const =0;
  };

  template <class T>
  class WorkerMaker : public Maker {
  public:
    //typedef T worker_type;
    explicit WorkerMaker();
  private:
    virtual void fillDescriptions(ConfigurationDescriptions& iDesc) const;
    virtual std::unique_ptr<Worker> makeWorker(WorkerParams const& p, ModuleDescription const& md, std::shared_ptr<maker::ModuleHolder> mod) const;
    virtual std::shared_ptr<maker::ModuleHolder> makeModule(edm::ParameterSet const& p) const;
    virtual const std::string& baseType() const;
  };

  template <class T>
  WorkerMaker<T>::WorkerMaker() {
  }

  template <class T>
  void WorkerMaker<T>::fillDescriptions(ConfigurationDescriptions& iDesc) const {
    T::fillDescriptions(iDesc);
    T::prevalidate(iDesc);
  }

  template<class T>
  std::shared_ptr<maker::ModuleHolder> WorkerMaker<T>::makeModule(edm::ParameterSet const& p) const
  {
    typedef T UserType;
    typedef typename UserType::ModuleType ModuleType;
    typedef typename UserType::WorkerType WorkerType;
    
    return std::shared_ptr<maker::ModuleHolder>(new maker::ModuleHolderT<ModuleType>{WorkerType::template makeModule<UserType>(p).release()});
  }
  
  template <class T>
  std::unique_ptr<Worker> WorkerMaker<T>::makeWorker(WorkerParams const& p, ModuleDescription const& md,
                                                     std::shared_ptr<maker::ModuleHolder> mod) const {
    typedef T UserType;
    typedef typename UserType::ModuleType ModuleType;
    typedef typename UserType::WorkerType WorkerType;

    maker::ModuleHolderT<ModuleType>* h = dynamic_cast<maker::ModuleHolderT<ModuleType>*>(mod.get());
    return std::unique_ptr<Worker>(new WorkerType(h->module(), md, p));
  }
  

  template<class T>
  const std::string& WorkerMaker<T>::baseType() const {
    return T::baseType();
  }
  
}

#endif
