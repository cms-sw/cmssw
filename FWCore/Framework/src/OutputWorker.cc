
/*----------------------------------------------------------------------
$Id: OutputWorker.cc,v 1.35 2008/01/11 20:30:08 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/Framework/src/OutputWorker.h"

namespace edm {
  OutputWorker::OutputWorker(std::auto_ptr<OutputModule> mod,
			     ModuleDescription const& md,
			     WorkerParams const& wp):
      WorkerT<OutputModule>(mod, md, wp)
  {
  }

  OutputWorker::~OutputWorker() {
  }

  void
  OutputWorker::maybeEndFile() {
    module().maybeEndFile();
  }

  void
  OutputWorker::closeFile() {
    module().doCloseFile();
  }

  void
  OutputWorker::openNewFileIfNeeded() {
    module().maybeOpenFile();
  }

  void
  OutputWorker::openFile(FileBlock const& fb) {
    module().doOpenFile(fb);
  }

  void
  OutputWorker::writeRun(RunPrincipal const& rp) {
    module().doWriteRun(rp);
  }

  void
  OutputWorker::writeLumi(LuminosityBlockPrincipal const& lbp) {
    module().doWriteLuminosityBlock(lbp);
  }

  bool 
  OutputWorker::implDoWork(EventPrincipal& ep, EventSetup const&,
			   BranchActionType,
			   CurrentProcessingContext const* cpc) {
    // EventSetup is not (yet) used. Should it be passed to the
    // OutputModule?
    bool rc = false;

    module().writeEvent(ep,description(), cpc);
    rc = true;
    return rc;
  }

  bool
  OutputWorker::implDoWork(RunPrincipal& rp, EventSetup const&,
			   BranchActionType bat,
			   CurrentProcessingContext const* cpc) {
    bool rc = false;
    if (bat == BranchActionBegin) module().doBeginRun(rp,description(),cpc);
    else module().doEndRun(rp,description(),cpc);
    rc = true;
    return rc;
  }


  bool
  OutputWorker::implDoWork(LuminosityBlockPrincipal& lbp, EventSetup const&,
			   BranchActionType bat,
			   CurrentProcessingContext const* cpc) {
    bool rc = false;
    if (bat == BranchActionBegin) module().doBeginLuminosityBlock(lbp,description(),cpc);
    else module().doEndLuminosityBlock(lbp,description(),cpc);
    rc = true;
    return rc;
  }

  void 
  OutputWorker::implBeginJob(EventSetup const& es) {
    module().selectProducts();
    module().doBeginJob(es);
  }

  std::string OutputWorker::workerType() const {
    return "OutputModule";
  }
  
  bool OutputWorker::wantAllEvents() const {return module().wantAllEvents();}

  bool OutputWorker::limitReached() const {return module().limitReached();}

  void OutputWorker::configure(OutputModuleDescription const& desc) {module().configure(desc);}
}
