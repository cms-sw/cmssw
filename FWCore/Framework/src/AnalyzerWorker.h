#ifndef Framework_AnalyzerWorker_h
#define Framework_AnalyzerWorker_h

/*----------------------------------------------------------------------
  
AnalyzerWorker: The EDAnalyzer as the schedule sees it.  The job of
this object is to call the producer, collect up the results, and
feed them into the event.
According to our current definition, a single producer can only
appear in one worker.

$Id: AnalyzerWorker.h,v 1.15 2005/12/28 00:21:58 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>

#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/src/Worker.h"

namespace edm
{
  class EDAnalyzer;
  class ModuleDescription;
  class WorkerParams;

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
    virtual bool implDoWork(EventPrincipal& e, EventSetup const& c);

    virtual void implBeginJob(EventSetup const&) ;
    virtual void implEndJob() ;
    virtual std::string workerType() const;
    
    boost::shared_ptr<EDAnalyzer> analyzer_;
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

#endif
