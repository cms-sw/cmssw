#ifndef FWCore_Framework_WorkerMaker_h
#define FWCore_Framework_WorkerMaker_h

#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/Framework/src/WorkerParams.h"

#include <memory>
#include <string>
#include "sigc++/signal.h"


namespace edm {
  class ConfigurationDescriptions;
  class ModuleDescription;
  class ParameterSet;
  
  class Maker {
  public:
    virtual ~Maker();
    std::auto_ptr<Worker> makeWorker(WorkerParams const&,
                                     sigc::signal<void, ModuleDescription const&>& iPre,
                                     sigc::signal<void, ModuleDescription const&>& iPost) const;
    void swapModule(Worker*, ParameterSet const&);
  protected:
    ModuleDescription createModuleDescription(WorkerParams const& p) const;

    void throwConfigurationException(ModuleDescription const& md,
                                     sigc::signal<void, ModuleDescription const&>& post,
                                     cms::Exception & iException) const;

    void throwValidationException(WorkerParams const& p,
				  cms::Exception & iException) const;

    void validateEDMType(std::string const& edmType, WorkerParams const& p) const;
                                             
  private:
    virtual void fillDescriptions(ConfigurationDescriptions& iDesc) const = 0;
    virtual std::auto_ptr<Worker> makeWorker(WorkerParams const& p, 
                                             ModuleDescription const& md) const = 0;
    virtual const std::string& baseType() const =0;
    virtual void implSwapModule(Worker*, ParameterSet const&)=0;
  };

  template <class T>
  class WorkerMaker : public Maker {
  public:
    //typedef T worker_type;
    explicit WorkerMaker();
  private:
    virtual void fillDescriptions(ConfigurationDescriptions& iDesc) const;
    virtual std::auto_ptr<Worker> makeWorker(WorkerParams const& p, ModuleDescription const& md) const;
    virtual const std::string& baseType() const;
    virtual void implSwapModule(Worker*, ParameterSet const&);
  };

  template <class T>
  WorkerMaker<T>::WorkerMaker() {
  }

  template <class T>
  void WorkerMaker<T>::fillDescriptions(ConfigurationDescriptions& iDesc) const {
    T::fillDescriptions(iDesc);
  }

  template <class T>
  std::auto_ptr<Worker> WorkerMaker<T>::makeWorker(WorkerParams const& p, ModuleDescription const& md) const {
    typedef T UserType;
    typedef typename UserType::ModuleType ModuleType;
    typedef typename UserType::WorkerType WorkerType;
    
    std::auto_ptr<ModuleType> module(WorkerType::template makeModule<UserType>(md, *p.pset_));    
    return std::auto_ptr<Worker>(new WorkerType(module, md, p));
  }
  

  template <class T>
  void WorkerMaker<T>::implSwapModule(Worker* w, ParameterSet const& p) {
    typedef T UserType;
    typedef typename UserType::ModuleType ModuleType;
    typedef typename UserType::WorkerType WorkerType;
          
    WorkerType* wt = dynamic_cast<WorkerType*>(w);
    assert(0!=wt);

    std::auto_ptr<ModuleType> module(WorkerType::template makeModule<UserType>(w->description(), p));
     
    wt->setModule(module);
  }
  
  template<class T>
  const std::string& WorkerMaker<T>::baseType() const {
    return T::baseType();
  }
  
}

#endif
