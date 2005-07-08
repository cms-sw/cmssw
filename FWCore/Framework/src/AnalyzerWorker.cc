
/*----------------------------------------------------------------------
$Id: AnalyzerWorker.cc,v 1.1 2005/05/29 02:29:53 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/CoreFramework/src/AnalyzerWorker.h"

#include "FWCore/CoreFramework/interface/EventPrincipal.h"
#include "FWCore/CoreFramework/interface/Event.h"
#include "FWCore/CoreFramework/interface/EDAnalyzer.h"

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
