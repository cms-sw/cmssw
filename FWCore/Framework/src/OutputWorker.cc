
/*----------------------------------------------------------------------
$Id: OutputWorker.cc,v 1.24 2007/07/09 07:29:51 llista Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/Framework/src/OutputWorker.h"

namespace edm {
  OutputWorker::OutputWorker(std::auto_ptr<OutputModule> mod,
			     ModuleDescription const& md,
			     WorkerParams const& wp):
      Worker(md,wp),
      mod_(mod)
  {
  }

  OutputWorker::~OutputWorker() {
  }

  void
  OutputWorker::maybeEndFile() {
    mod_->maybeEndFile();
  }

  void
  OutputWorker::doEndFile() {
    mod_->doEndFile();
  }


  bool 
  OutputWorker::implDoWork(EventPrincipal& ep, EventSetup const&,
			   BranchActionType,
			   CurrentProcessingContext const* cpc) {
    // EventSetup is not (yet) used. Should it be passed to the
    // OutputModule?
    bool rc = false;

    mod_->writeEvent(ep,description(), cpc);
    rc = true;
    return rc;
  }

  bool
  OutputWorker::implDoWork(RunPrincipal& rp, EventSetup const&,
			   BranchActionType bat,
			   CurrentProcessingContext const* cpc) {
    bool rc = false;
    if (bat == BranchActionBegin) mod_->doBeginRun(rp,description(),cpc);
    else mod_->doEndRun(rp,description(),cpc);
    rc = true;
    return rc;
  }


  bool
  OutputWorker::implDoWork(LuminosityBlockPrincipal& lbp, EventSetup const&,
			   BranchActionType bat,
			   CurrentProcessingContext const* cpc) {
    bool rc = false;
    if (bat == BranchActionBegin) mod_->doBeginLuminosityBlock(lbp,description(),cpc);
    else mod_->doEndLuminosityBlock(lbp,description(),cpc);
    rc = true;
    return rc;
  }

  void 
  OutputWorker::implBeginJob(EventSetup const& es) {
    mod_->selectProducts();
    mod_->doBeginJob(es);
  }

  void 
  OutputWorker::implEndJob() {
    mod_->doEndJob();
  }

  std::string OutputWorker::workerType() const {
    return "OutputModule";
  }
  
  int OutputWorker::eventCount() const {
    return mod_->eventCount();
  }
}
