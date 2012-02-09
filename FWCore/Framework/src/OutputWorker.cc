
/*----------------------------------------------------------------------
$Id: OutputWorker.cc,v 1.38 2008/07/29 02:17:36 wmtan Exp $
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
  OutputWorker::closeFile() {
    module().doCloseFile();
  }

  bool
  OutputWorker::shouldWeCloseFile() const {
    return module().shouldWeCloseFile();
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

  bool OutputWorker::wantAllEvents() const {return module().wantAllEvents();}

  bool OutputWorker::limitReached() const {return module().limitReached();}

  void OutputWorker::configure(OutputModuleDescription const& desc) {module().configure(desc);}
  
  SelectionsArray const& OutputWorker::keptProducts() const {
    return module().keptProducts();
  }
}
