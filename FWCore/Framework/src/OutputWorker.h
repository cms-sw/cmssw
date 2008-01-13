#ifndef Framework_OutputWorker_h
#define Framework_OutputWorker_h

/*----------------------------------------------------------------------
  
OutputWorker: The OutputModule as the schedule sees it.  The job of
this object is to call the output module.

According to our current definition, a single output module can only
appear in one worker.

$Id: OutputWorker.h,v 1.32 2008/01/11 20:30:08 wmtan Exp $
----------------------------------------------------------------------*/

#include <memory>

#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {

  class OutputWorker : public WorkerT<OutputModule> {
  public:
    OutputWorker(std::auto_ptr<OutputModule> mod, 
		 ModuleDescription const&,
		 WorkerParams const&);

    virtual ~OutputWorker();

    // Call maybeEndFile() on the controlled OutputModule.
    void maybeEndFile();

    // Call closeFile() on the controlled OutputModule.
    void closeFile();

    void openNewFileIfNeeded();

    bool wantAllEvents() const;

    void openFile(FileBlock const& fb);

    void writeRun(RunPrincipal const& rp);

    void writeLumi(LuminosityBlockPrincipal const& lbp);

    bool limitReached() const;

    void configure(OutputModuleDescription const& desc);

  private:
    virtual bool implDoWork(EventPrincipal& e, EventSetup const& c,
			    BranchActionType,
			    CurrentProcessingContext const* cpc);
    virtual bool implDoWork(RunPrincipal& rp, EventSetup const& c,
			    BranchActionType bat,
			    CurrentProcessingContext const* cpc);
    virtual bool implDoWork(LuminosityBlockPrincipal& lbp, EventSetup const& c,
			    BranchActionType bat,
			    CurrentProcessingContext const* cpc);

    virtual void implBeginJob(EventSetup const& es);

    virtual std::string workerType() const;
  };

  template <> 
  struct WorkerType<OutputModule> {
    typedef OutputModule ModuleType;
    typedef OutputWorker worker_type;
  };

}

#endif
