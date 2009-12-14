#ifndef FWCore_Framework_WorkerMaker_h
#define FWCore_Framework_WorkerMaker_h

#include "FWCore/Framework/src/WorkerT.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <memory>
#include <string>
#include "sigc++/signal.h"


namespace edm {
  
  class Maker {
  public:
    virtual ~Maker();
    virtual std::auto_ptr<Worker> makeWorker(WorkerParams const&,
                                             sigc::signal<void, ModuleDescription const&>& iPre,
                                             sigc::signal<void, ModuleDescription const&>& iPost) const = 0;
  protected:
    ModuleDescription createModuleDescription(WorkerParams const &p) const;

    void throwConfigurationException(ModuleDescription const &md,
                                     sigc::signal<void, ModuleDescription const&>& post,
                                     cms::Exception const& iException) const;

    void throwValidationException(WorkerParams const& p,
				  cms::Exception const& iException) const;

    void validateEDMType(const std::string & edmType, WorkerParams const& p) const;
  };

  template <class T>
  class WorkerMaker : public Maker {
  public:
    //typedef T worker_type;
    explicit WorkerMaker();
    virtual std::auto_ptr<Worker> makeWorker(WorkerParams const&,
                                     sigc::signal<void, ModuleDescription const&>&,
                                     sigc::signal<void, ModuleDescription const&>&) const;
  };

  template <class T>
  WorkerMaker<T>::WorkerMaker() {
  }

  template <class T>
  std::auto_ptr<Worker> WorkerMaker<T>::makeWorker(WorkerParams const& p,
                                                   sigc::signal<void, ModuleDescription const&>& pre,
                                                   sigc::signal<void, ModuleDescription const&>& post) const {
    typedef T UserType;
    typedef typename UserType::ModuleType ModuleType;
    typedef typename UserType::WorkerType WorkerType;

    try {
      ConfigurationDescriptions descriptions;
      UserType::fillDescriptions(descriptions);
      descriptions.validate(*p.pset_, p.pset_->getParameter<std::string>("@module_label"));
      p.pset_->registerIt();
    }
    catch (cms::Exception& iException) {
      throwValidationException(p, iException);
    }

    ModuleDescription md = createModuleDescription(p);

    std::auto_ptr<Worker> worker;
    try {
       pre(md);

       std::auto_ptr<ModuleType> module(WorkerType::template makeModule<UserType>(md, *p.pset_));
       validateEDMType(module->baseType(), p);

       worker=std::auto_ptr<Worker>(new WorkerType(module, md, p));
       post(md);
    } catch( cms::Exception& iException){
       throwConfigurationException(md, post, iException);
    }
    return worker;
  }

}

#endif
