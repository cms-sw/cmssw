
/*----------------------------------------------------------------------
$Id: AnalyzerWorker.cc,v 1.17 2008/01/11 20:30:08 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/src/AnalyzerWorker.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <iostream>

namespace edm {
  AnalyzerWorker::AnalyzerWorker(std::auto_ptr<EDAnalyzer> ed,
				 ModuleDescription const& md,
				 WorkerParams const& wp):
    WorkerT<EDAnalyzer>(ed, md, wp) {
  }

  AnalyzerWorker::~AnalyzerWorker() {
  }

  bool AnalyzerWorker::implDoWork(EventPrincipal& ep, EventSetup const& c,
				  BranchActionType,
				  CurrentProcessingContext const* cpc) {
    bool rc = false;
    Event e(ep,description());
    module().doAnalyze(e,c,cpc);
    rc = true;
    return rc;
  }

  bool AnalyzerWorker::implDoWork(RunPrincipal& rp, EventSetup const& c,
				  BranchActionType bat,
				  CurrentProcessingContext const* cpc) {
    bool rc = false;
    Run r(rp,description());
    if (bat == BranchActionBegin) module().doBeginRun(r,c,cpc);
    else module().doEndRun(r,c,cpc);
    rc = true;
    return rc;
  }

  bool AnalyzerWorker::implDoWork(LuminosityBlockPrincipal& lbp, EventSetup const& c,
				  BranchActionType bat,
				  CurrentProcessingContext const* cpc) {
    bool rc = false;
    LuminosityBlock lb(lbp,description());
    if (bat == BranchActionBegin) module().doBeginLuminosityBlock(lb,c,cpc);
    else module().doEndLuminosityBlock(lb,c,cpc);
    rc = true;
    return rc;
  }

  std::string AnalyzerWorker::workerType() const {
    return "EDAnalyzer";
  }
}
