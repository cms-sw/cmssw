#ifndef EDM_EDANALYZERWORKER_INCLUDED
#define EDM_EDANALYZERWORKER_INCLUDED

/*----------------------------------------------------------------------
  
AnalyzerWorker: The EDAnalyzer as the schedule sees it.  The job of
this object is to call the producer, collect up the results, and
feed them into the event.
According to our current definition, a single producer can only
appear in one worker.

$Id: AnalyzerWorker.h,v 1.5 2005/07/14 22:50:53 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>

#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/interface/ModuleDescription.h"

namespace edm
{
  class ActionTable;
  class WorkerParams;

  class AnalyzerWorker : public Worker
  {
  public:
    AnalyzerWorker(std::auto_ptr<EDAnalyzer>,
		   const ModuleDescription&,
		   const WorkerParams&);
    virtual ~AnalyzerWorker();

  private:
    virtual bool doWork(EventPrincipal& e, EventSetup const& c);

    virtual void beginJob( EventSetup const& ) ;
    virtual void endJob() ;
    
    ModuleDescription md_;
    boost::shared_ptr<EDAnalyzer> analyzer_;
    const ActionTable* actions_; // memory assumed to be managed elsewhere
  };

  template <> 
  struct WorkerType<EDAnalyzer>
  {
    typedef EDAnalyzer ModuleType;
    typedef AnalyzerWorker worker_type;
  };

}

#endif // EDM_EDANALYZERWORKER_INCLUDED
