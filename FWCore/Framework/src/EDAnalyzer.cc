/*----------------------------------------------------------------------
  
$Id: EDAnalyzer.cc,v 1.5 2006/06/20 23:13:27 paterno Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/Framework/src/CPCSentry.h"

namespace edm
{
  EDAnalyzer::~EDAnalyzer()
  { }

  void
  EDAnalyzer::doAnalyze(Event const& e, EventSetup const& c,
			CurrentProcessingContext const* cpc)
  {
    detail::CPCSentry sentry(current_context_, cpc);
    this->analyze(e,c);
  }

  void 
  EDAnalyzer::doBeginJob(EventSetup const& es)
  {
    this->beginJob(es);
  }
  
  void 
  EDAnalyzer::doEndJob()
  {
    this->endJob();
  }

  void 
  EDAnalyzer::beginJob(EventSetup const&) 
  { }
   
  void 
  EDAnalyzer::endJob()
  { }

  CurrentProcessingContext const*
  EDAnalyzer::currentContext() const
  {
    return current_context_;
  }

}
  
