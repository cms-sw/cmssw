#ifndef WORKERMAKER_H
#define WORKERMAKER_H

#include "FWCore/CoreFramework/src/Worker.h"
#include "FWCore/CoreFramework/interface/Provenance.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// The following includes are temporary until a better
// solution can be found.  Placing these includes here
// leads to more physical coupling than is probably necessary.
// Another solution is to build a typeid lookup table in the 
// implementation file (one every for each XXXWorker) and
// then include all the relevent worker headers in the 
// implementation file only.
#include "FWCore/CoreFramework/src/ProducerWorker.h"
#include "FWCore/CoreFramework/src/FilterWorker.h"
#include "FWCore/CoreFramework/src/AnalyzerWorker.h"
#include "FWCore/CoreFramework/src/OutputWorker.h"

#include <memory>
#include <string>

namespace edm {
  
  class Maker
  {
  public:
    virtual ~Maker();
    virtual std::auto_ptr<Worker> makeWorker(ParameterSet const&,
					     std::string const& pn,
					     unsigned long vn,
					     unsigned long pass
					     ) const = 0;
  };

  template <class T>
  class WorkerMaker : public Maker
  {
  public:
    typedef T worker_type;
    explicit WorkerMaker();
    std::auto_ptr<Worker> makeWorker(ParameterSet const& conf,
				     std::string const& pn,
				     unsigned long vn,
				     unsigned long pass) const;
  };

  template <class T>
  WorkerMaker<T>::WorkerMaker()
  {
  }

  template <class T>
  std::auto_ptr<Worker> WorkerMaker<T>::makeWorker(ParameterSet const& conf,
						   std::string const& pn,
						   unsigned long vn,
						   unsigned long pass) const
  {
    typedef T user_type;
    typedef typename user_type::module_type module_type;
    typedef typename WorkerType<module_type>::worker_type  worker_type;

    ModuleDescription md;
    md.pid = PS_ID("oink"); // conf.id();
    md.module_name = conf.getString("module_type");
    md.module_label = conf.getString("module_label");
    md.version_number = vn;
    md.process_name = pn;
    md.pass = pass; 

    std::auto_ptr<module_type> module(new user_type(conf));
    std::auto_ptr<Worker> worker(new worker_type(module, md));
    return worker;
  }

}

#endif
