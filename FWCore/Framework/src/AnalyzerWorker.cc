
/*----------------------------------------------------------------------
$Id: AnalyzerWorker.cc,v 1.16 2007/06/29 03:43:21 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/src/AnalyzerWorker.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/src/WorkerParams.h"

#include <iostream>

namespace edm {
  AnalyzerWorker::AnalyzerWorker(std::auto_ptr<EDAnalyzer> ed,
				 ModuleDescription const& md,
				 WorkerParams const& wp):
    Worker(md,wp),
    analyzer_(ed)
  {
  }

  AnalyzerWorker::~AnalyzerWorker() {
  }

  bool AnalyzerWorker::implDoWork(EventPrincipal& ep, EventSetup const& c,
				  BranchActionType,
				  CurrentProcessingContext const* cpc) {
    bool rc = false;
    Event e(ep,description());
    analyzer_->doAnalyze(e,c,cpc);
    rc = true;
    return rc;
  }

  bool AnalyzerWorker::implDoWork(RunPrincipal& rp, EventSetup const& c,
				  BranchActionType bat,
				  CurrentProcessingContext const* cpc) {
    bool rc = false;
    Run r(rp,description());
    if (bat == BranchActionBegin) analyzer_->doBeginRun(r,c,cpc);
    else analyzer_->doEndRun(r,c,cpc);
    rc = true;
    return rc;
  }

  bool AnalyzerWorker::implDoWork(LuminosityBlockPrincipal& lbp, EventSetup const& c,
				  BranchActionType bat,
				  CurrentProcessingContext const* cpc) {
    bool rc = false;
    LuminosityBlock lb(lbp,description());
    if (bat == BranchActionBegin) analyzer_->doBeginLuminosityBlock(lb,c,cpc);
    else analyzer_->doEndLuminosityBlock(lb,c,cpc);
    rc = true;
    return rc;
  }

  void AnalyzerWorker::implBeginJob(EventSetup const& es) {
    analyzer_->doBeginJob(es);
  }

  void AnalyzerWorker::implEndJob() {
    analyzer_->doEndJob();
  }
  
  void AnalyzerWorker::implRespondToOpenInputFile(FileBlock const& fb) {
    analyzer_->doRespondToOpenInputFile(fb);
  }

  void AnalyzerWorker::implRespondToCloseInputFile(FileBlock const& fb) {
    analyzer_->doRespondToCloseInputFile(fb);
  }

  void AnalyzerWorker::implRespondToOpenOutputFiles(FileBlock const& fb) {
    analyzer_->doRespondToOpenOutputFiles(fb);
  }

  void AnalyzerWorker::implRespondToCloseOutputFiles(FileBlock const& fb) {
    analyzer_->doRespondToCloseOutputFiles(fb);
  }

  std::string AnalyzerWorker::workerType() const {
    return "EDAnalyzer";
  }
}
