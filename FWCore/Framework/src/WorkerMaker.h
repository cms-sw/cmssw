#ifndef FWCore_Framework_WorkerMaker_h
#define FWCore_Framework_WorkerMaker_h

#include <cassert>
#include <memory>
#include <string>

#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/Framework/src/MakeModuleParams.h"
#include "FWCore/Framework/src/ModuleHolder.h"
#include "FWCore/Framework/src/MakeModuleHelper.h"

#include "FWCore/Utilities/interface/Signal.h"


namespace edm {
  class ConfigurationDescriptions;
  class ModuleDescription;
  class ParameterSet;
  class Maker;
  class ExceptionToActionTable;
  
  class Maker {
  public:
    virtual ~Maker();
    std::shared_ptr<maker::ModuleHolder> makeModule(MakeModuleParams const&,
                                       signalslot::Signal<void(ModuleDescription const&)>& iPre,
                                       signalslot::Signal<void(ModuleDescription const&)>& iPost) const;
    std::unique_ptr<Worker> makeWorker(ExceptionToActionTable const*,
                                       maker::ModuleHolder const*) const;

    std::shared_ptr<maker::ModuleHolder> makeReplacementModule(edm::ParameterSet const& p) const { return makeModule(p);}
protected:
      
    ModuleDescription createModuleDescription(MakeModuleParams const& p) const;

    void throwConfigurationException(ModuleDescription const& md,
                                     signalslot::Signal<void(ModuleDescription const&)>& post,
                                     cms::Exception & iException) const;

    void throwValidationException(MakeModuleParams const& p,
				  cms::Exception & iException) const;

    void validateEDMType(std::string const& edmType, MakeModuleParams const& p) const;

  private:
    virtual void fillDescriptions(ConfigurationDescriptions& iDesc) const = 0;
    virtual std::shared_ptr<maker::ModuleHolder> makeModule(edm::ParameterSet const& p) const  = 0;
    virtual std::unique_ptr<Worker> makeWorker(ExceptionToActionTable const* actions,
                                             ModuleDescription const& md,
                                               maker::ModuleHolder const* mod) const = 0;
    virtual const std::string& baseType() const =0;
  };
  
  

  template <class T>
  class WorkerMaker : public Maker {
  public:
    //typedef T worker_type;
    explicit WorkerMaker();
  private:
    virtual void fillDescriptions(ConfigurationDescriptions& iDesc) const;
    virtual std::unique_ptr<Worker> makeWorker(ExceptionToActionTable const* actions, ModuleDescription const& md, maker::ModuleHolder const* mod) const;
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
    typedef MakeModuleHelper<ModuleType> MakerHelperType;
    
    
    
    return std::shared_ptr<maker::ModuleHolder>(new maker::ModuleHolderT<ModuleType>{MakerHelperType::template makeModule<UserType>(p).release(),this});
  }
  
  template <class T>
  std::unique_ptr<Worker> WorkerMaker<T>::makeWorker(ExceptionToActionTable const* actions, ModuleDescription const& md,
                                                     maker::ModuleHolder const* mod) const {
    typedef T UserType;
    typedef typename UserType::ModuleType ModuleType;
    typedef edm::WorkerT<ModuleType> WorkerType;

    maker::ModuleHolderT<ModuleType> const* h = dynamic_cast<maker::ModuleHolderT<ModuleType> const*>(mod);
    return std::unique_ptr<Worker>(new WorkerType(h->module(), md, actions));
  }
  

  template<class T>
  const std::string& WorkerMaker<T>::baseType() const {
    return T::baseType();
  }
  
}

#endif
