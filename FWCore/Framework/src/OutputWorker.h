#ifndef Framework_OutputWorker_h
#define Framework_OutputWorker_h

/*----------------------------------------------------------------------
  
OutputWorker: The OutputModule as the schedule sees it.  The job of
this object is to call the output module.

According to our current definition, a single output module can only
appear in one worker.

$Id: OutputWorker.h,v 1.17 2006/06/20 23:13:27 paterno Exp $
----------------------------------------------------------------------*/

#include <memory>

#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm
{

  class OutputWorker : public Worker
  {
  public:
    OutputWorker(std::auto_ptr<OutputModule> mod, 
		 const ModuleDescription&,
		 const WorkerParams&);

    virtual ~OutputWorker();

    template <class ModType>
    static std::auto_ptr<OutputModule> makeOne(const ModuleDescription& md,
					const WorkerParams& wp);
  private:
    virtual bool implDoWork(EventPrincipal& e, EventSetup const& c,
			    CurrentProcessingContext const* cpc);

    virtual void implBeginJob(EventSetup const&) ;
    virtual void implEndJob() ;
    virtual bool implBeginRun(RunPrincipal& rp, EventSetup const& c,
			    CurrentProcessingContext const* cpc);
    virtual bool implEndRun(RunPrincipal& rp, EventSetup const& c,
			    CurrentProcessingContext const* cpc);
    virtual bool implBeginLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
			    CurrentProcessingContext const* cpc);
    virtual bool implEndLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
			    CurrentProcessingContext const* cpc);
    virtual std::string workerType() const;
    
    boost::shared_ptr<OutputModule> mod_;
  };

  template <> 
  struct WorkerType<OutputModule>
  {
    typedef OutputModule ModuleType;
    typedef OutputWorker worker_type;
  };

  template <class ModType>
  std::auto_ptr<OutputModule> OutputWorker::makeOne(const ModuleDescription&,
						    const WorkerParams& wp)
  {
    return std::auto_ptr<OutputModule>(new ModType(*wp.pset_));
  }

}

#endif
