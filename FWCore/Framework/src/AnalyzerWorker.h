#ifndef EDM_EDANALYZERWORKER_INCLUDED
#define EDM_EDANALYZERWORKER_INCLUDED

/*----------------------------------------------------------------------
  
AnalyzerWorker: The EDAnalyzer as the schedule sees it.  The job of
this object is to call the producer, collect up the results, and
feed them into the event.
According to our current definition, a single producer can only
appear in one worker.

$Id: AnalyzerWorker.h,v 1.2 2005/04/21 04:21:38 jbk Exp $

----------------------------------------------------------------------*/

#include <memory>

#include "boost/shared_ptr.hpp"

#include "FWCore/CoreFramework/interface/CoreFrameworkfwd.h"
#include "FWCore/CoreFramework/src/Worker.h"
#include "FWCore/CoreFramework/interface/Provenance.h"

namespace edm
{

  class AnalyzerWorker : public Worker
  {
  public:
    AnalyzerWorker(std::auto_ptr<EDAnalyzer>, const ModuleDescription&);
    virtual ~AnalyzerWorker();

  private:
    virtual bool doWork(EventPrincipal& e, EventSetup const& c);

    ModuleDescription md_;
    boost::shared_ptr<EDAnalyzer> analyzer_;
  };

  template <> 
  struct WorkerType<EDAnalyzer>
  {
    typedef EDAnalyzer module_type;
    typedef AnalyzerWorker worker_type;
  };

}

#endif // EDM_EDANALYZERWORKER_INCLUDED
