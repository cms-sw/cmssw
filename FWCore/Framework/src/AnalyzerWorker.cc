
/*----------------------------------------------------------------------
$Id: AnalyzerWorker.cc,v 1.11 2006/02/08 00:44:25 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/src/AnalyzerWorker.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/ModuleDescription.h"
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
  
  std::string AnalyzerWorker::workerType() const {
    return "EDAnalyzer";
  }
}
