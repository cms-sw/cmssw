
/*----------------------------------------------------------------------
$Id: AnalyzerWorker.cc,v 1.13 2006/11/03 17:57:52 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/src/AnalyzerWorker.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/Framework/src/CPCSentry.h"


#include <iostream>

using namespace std;

namespace edm
{
  AnalyzerWorker::AnalyzerWorker(std::auto_ptr<EDAnalyzer> ed,
				 const ModuleDescription& md,
				 const WorkerParams& wp):
    Worker(md,wp),
    analyzer_(ed)
  {
  }

  AnalyzerWorker::~AnalyzerWorker()
  {
  }

  bool AnalyzerWorker::implDoWork(EventPrincipal& ep, EventSetup const& c,
				  CurrentProcessingContext const* cpc)
  {
    bool rc = false;
    Event e(ep,description());
    analyzer_->doAnalyze(e,c,cpc);
    rc = true;
    return rc;
  }

  void AnalyzerWorker::implBeginJob(EventSetup const& es) 
  {
    analyzer_->doBeginJob(es);
  }

  void AnalyzerWorker::implEndJob() 
  {
    analyzer_->doEndJob();
  }
  
  bool AnalyzerWorker::implBeginRun(RunPrincipal& rp, EventSetup const& c,
				  CurrentProcessingContext const* cpc)
  {
    bool rc = false;
    Run r(rp,description());
    analyzer_->doBeginRun(r,c,cpc);
    rc = true;
    return rc;
  }

  bool AnalyzerWorker::implEndRun(RunPrincipal& rp, EventSetup const& c,
				  CurrentProcessingContext const* cpc)
  {
    bool rc = false;
    Run r(rp,description());
    analyzer_->doEndRun(r,c,cpc);
    rc = true;
    return rc;
  }

  bool AnalyzerWorker::implBeginLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
				  CurrentProcessingContext const* cpc)
  {
    bool rc = false;
    LuminosityBlock lb(lbp,description());
    analyzer_->doBeginLuminosityBlock(lb,c,cpc);
    rc = true;
    return rc;
  }

  bool AnalyzerWorker::implEndLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
				  CurrentProcessingContext const* cpc)
  {
    bool rc = false;
    LuminosityBlock lb(lbp,description());
    analyzer_->doEndLuminosityBlock(lb,c,cpc);
    rc = true;
    return rc;
  }

  std::string AnalyzerWorker::workerType() const {
    return "EDAnalyzer";
  }
}
