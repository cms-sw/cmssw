/*----------------------------------------------------------------------
  
$Id: EDAnalyzer.cc,v 1.6 2006/06/21 19:03:13 paterno Exp $

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
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    this->analyze(e, c);
  }

  void 
  EDAnalyzer::doBeginJob(EventSetup const& es) {
    this->beginJob(es);
  }
  
  void 
  EDAnalyzer::doEndJob() {
    this->endJob();
  }

  void
  EDAnalyzer::doBeginRun(Run const& r, EventSetup const& c,
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    this->beginRun(r, c);
  }

  void
  EDAnalyzer::doEndRun(Run const& r, EventSetup const& c,
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    this->endRun(r, c);
  }

  void
  EDAnalyzer::doBeginLuminosityBlock(LuminosityBlock const& lb, EventSetup const& c,
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    this->beginLuminosityBlock(lb, c);
  }

  void
  EDAnalyzer::doEndLuminosityBlock(LuminosityBlock const& lb, EventSetup const& c,
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    this->endLuminosityBlock(lb, c);
  }

  CurrentProcessingContext const*
  EDAnalyzer::currentContext() const
  {
    return current_context_;
  }

}
  
