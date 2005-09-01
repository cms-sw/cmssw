#ifndef Framework_EDAnalyzerWorker_h
#define Framework_EDAnalyzerWorker_h

/*----------------------------------------------------------------------
  
AnalyzerWorker: The EDAnalyzer as the schedule sees it.  The job of
this object is to call the producer, collect up the results, and
feed them into the event.
According to our current definition, a single producer can only
appear in one worker.

$Id: AnalyzerWorker.h,v 1.9 2005/08/25 20:24:52 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>

#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/Framework/interface/ModuleDescription.h"

namespace edm
{
  class ActionTable;
  class ParameterSet;

  class AnalyzerWorker : public Worker
  {
  public:
    AnalyzerWorker(std::auto_ptr<EDAnalyzer>,
		   const ModuleDescription&,
		   const WorkerParams&);
    virtual ~AnalyzerWorker();

  template <class ModType>
  static std::auto_ptr<EDAnalyzer> makeOne(const ModuleDescription& md,
					   const WorkerParams& wp);

  private:
    virtual bool doWork(EventPrincipal& e, EventSetup const& c);

    virtual void beginJob(EventSetup const&) ;
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

  template <class ModType>
  std::auto_ptr<EDAnalyzer> AnalyzerWorker::makeOne(const ModuleDescription&,
						    const WorkerParams& wp)
  {
    return std::auto_ptr<EDAnalyzer>(new ModType(*wp.pset_));
  }
}

#endif // Framework_EDAnalyzerWorker_h
