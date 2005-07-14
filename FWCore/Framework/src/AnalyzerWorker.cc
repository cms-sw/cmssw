
/*----------------------------------------------------------------------
$Id: AnalyzerWorker.cc,v 1.2 2005/07/08 00:09:42 chrjones Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/src/AnalyzerWorker.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

namespace edm
{
  AnalyzerWorker::AnalyzerWorker(std::auto_ptr<EDAnalyzer> ed,
				 const ModuleDescription& md):
    md_(md),
    analyzer_(ed)
  {
  }

  AnalyzerWorker::~AnalyzerWorker()
  {
  }

  bool AnalyzerWorker::doWork(EventPrincipal& ep, EventSetup const& c)
  {
    Event e(ep,md_);
    analyzer_->analyze(e,c);
    return true;
  }

  void AnalyzerWorker::beginJob( EventSetup const& es) 
  {
    analyzer_->beginJob(es);
  }

  void AnalyzerWorker::endJob() 
  {
    analyzer_->endJob();
  }
}
