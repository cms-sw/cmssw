
/*----------------------------------------------------------------------
$Id: AnalyzerWorker.cc,v 1.10 2005/12/28 00:21:58 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/src/AnalyzerWorker.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/Utilities/interface/Exception.h"

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

  bool AnalyzerWorker::implDoWork(EventPrincipal& ep, EventSetup const& c)
  {
    bool rc = false;
    Event e(ep,description());
    analyzer_->analyze(e,c);
    rc = true;
    return rc;
  }

  void AnalyzerWorker::implBeginJob(EventSetup const& es) 
  {
    analyzer_->beginJob(es);
  }

  void AnalyzerWorker::implEndJob() 
  {
    analyzer_->endJob();
  }
  
  std::string AnalyzerWorker::workerType() const {
    return "EDAnalyzer";
  }
}
