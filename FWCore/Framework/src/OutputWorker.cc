
/*----------------------------------------------------------------------
$Id: OutputWorker.cc,v 1.17 2006/06/20 23:13:27 paterno Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/interface/Actions.h"
#include "DataFormats/Common/interface/ModuleDescription.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/Framework/src/OutputWorker.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>

using namespace std;

namespace edm {
  OutputWorker::OutputWorker(std::auto_ptr<OutputModule> mod,
			     ModuleDescription const& md,
			     WorkerParams const& wp):
      Worker(md,wp),
      mod_(mod) {
  }

  OutputWorker::~OutputWorker() {
  }

  bool 
  OutputWorker::implDoWork(EventPrincipal& ep, EventSetup const&,
			   CurrentProcessingContext const* cpc) {
    // EventSetup is not (yet) used. Should it be passed to the
    // OutputModule?
    bool rc = false;

    mod_->writeEvent(ep,description(), cpc);
    rc=true;
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

  bool OutputWorker::implBeginRun(RunPrincipal& rp, EventSetup const&,
				  CurrentProcessingContext const* cpc)
  {
    bool rc = false;
    mod_->doBeginRun(rp,description(),cpc);
    rc = true;
    return rc;
  }

  bool OutputWorker::implEndRun(RunPrincipal& rp, EventSetup const&,
				  CurrentProcessingContext const* cpc)
  {
    bool rc = false;
    mod_->doEndRun(rp,description(),cpc);
    rc = true;
    return rc;
  }

  bool OutputWorker::implBeginLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const&,
				  CurrentProcessingContext const* cpc)
  {
    bool rc = false;
    mod_->doBeginLuminosityBlock(lbp,description(),cpc);
    rc = true;
    return rc;
  }

  bool OutputWorker::implEndLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const&,
				  CurrentProcessingContext const* cpc)
  {
    bool rc = false;
    mod_->doEndLuminosityBlock(lbp,description(),cpc);
    rc = true;
    return rc;
  }


  std::string OutputWorker::workerType() const {
    return "OutputModule";
  }
  
}
