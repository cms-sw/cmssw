#ifndef Framework_WorkerMaker_h
#define Framework_WorkerMaker_h

#include "FWCore/Framework/src/Worker.h"
#include "DataFormats/Common/interface/ModuleDescription.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

// The following includes are temporary until a better
// solution can be found.  Placing these includes here
// leads to more physical coupling than is probably necessary.
// Another solution is to build a typeid lookup table in the 
// implementation file (one every for each XXXWorker) and
// then include all the relevent worker headers in the 
// implementation file only.
#include "FWCore/Framework/src/ProducerWorker.h"
#include "FWCore/Framework/src/FilterWorker.h"
#include "FWCore/Framework/src/AnalyzerWorker.h"
#include "FWCore/Framework/src/OutputWorker.h"

#include <memory>
#include <string>
#include "sigc++/signal.h"


namespace edm {
  
  class Maker
  {
  public:
    virtual ~Maker();
    virtual std::auto_ptr<Worker> makeWorker(const WorkerParams&,
                                             sigc::signal<void, const ModuleDescription&>& iPre,
                                             sigc::signal<void, const ModuleDescription&>& iPost) const = 0;
  };

  template <class T>
  class WorkerMaker : public Maker
  {
  public:
    typedef T worker_type;
    explicit WorkerMaker();
    std::auto_ptr<Worker> makeWorker(const WorkerParams&,
                                     sigc::signal<void, const ModuleDescription&>&,
                                     sigc::signal<void, const ModuleDescription&>&) const;
  };

  template <class T>
  WorkerMaker<T>::WorkerMaker()
  {
  }

  template <class T>
  std::auto_ptr<Worker> WorkerMaker<T>::makeWorker(const WorkerParams& p,
                                                   sigc::signal<void, const ModuleDescription&>& pre,
                                                   sigc::signal<void, const ModuleDescription&>& post) const
  {
    typedef T UserType;
    typedef typename UserType::ModuleType ModuleType;
    typedef typename WorkerType<ModuleType>::worker_type  worker_type;

    const ParameterSet& procParams = *p.procPset_;
    const ParameterSet& conf = *p.pset_;
    ModuleDescription md;
    md.parameterSetID_ = conf.id();
    md.moduleName_ = conf.template getParameter<std::string>("@module_type");
    md.moduleLabel_ = conf.template getParameter<std::string>("@module_label");
    md.processConfiguration_ = ProcessConfiguration(p.processName_, procParams.id(), p.releaseVersion_, p.passID_); 

    std::auto_ptr<Worker> worker;
    try {
       pre(md);
       std::auto_ptr<ModuleType> module(worker_type::template makeOne<UserType>(md,p));
       worker=std::auto_ptr<Worker>(new worker_type(module, md, p));
       post(md);
    } catch( cms::Exception& iException){
       edm::Exception toThrow(edm::errors::Configuration,"Error occured while creating ");
       toThrow<<md.moduleName_<<" with label "<<md.moduleLabel_<<"\n";
       toThrow.append(iException);
       post(md);
       throw toThrow;
    }
    return worker;
  }

}

#endif
