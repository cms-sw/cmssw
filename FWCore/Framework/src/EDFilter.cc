/*----------------------------------------------------------------------
  
$Id: EDFilter.cc,v 1.6 2006/06/21 19:03:13 paterno Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/Framework/src/CPCSentry.h"

namespace edm
{
  EDFilter::~EDFilter()
  { }

  bool
  EDFilter::doFilter(Event& e, EventSetup const& c,
		     CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    return this->filter(e, c);
  }

  void 
  EDFilter::doBeginJob(EventSetup const& es) { 
    this->beginJob(es);
  }
   
  void EDFilter::doEndJob() { 
    this->endJob();
  }

  bool
  EDFilter::doBeginRun(Run & r, EventSetup const& c,
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    return this->beginRun(r, c);
  }

  bool
  EDFilter::doEndRun(Run & r, EventSetup const& c,
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    return this->endRun(r, c);
  }

  bool
  EDFilter::doBeginLuminosityBlock(LuminosityBlock & lb, EventSetup const& c,
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    return this->beginLuminosityBlock(lb, c);
  }

  bool
  EDFilter::doEndLuminosityBlock(LuminosityBlock & lb, EventSetup const& c,
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    return this->endLuminosityBlock(lb, c);
  }

  CurrentProcessingContext const*
  EDFilter::currentContext() const
  {
    return current_context_;
  }
  
   
}
  
