/*----------------------------------------------------------------------
  
$Id: EDFilter.cc,v 1.9 2007/09/18 18:06:47 chrjones Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/src/CPCSentry.h"

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

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

  void
  EDFilter::doRespondToOpenInputFile(FileBlock const& fb) {
    respondToOpenInputFile(fb);
  }

  void
  EDFilter::doRespondToCloseInputFile(FileBlock const& fb) {
    respondToCloseInputFile(fb);
  }

  void
  EDFilter::doRespondToOpenOutputFiles(FileBlock const& fb) {
    respondToOpenOutputFiles(fb);
  }

  void
  EDFilter::doRespondToCloseOutputFiles(FileBlock const& fb) {
    respondToCloseOutputFiles(fb);
  }

  CurrentProcessingContext const*
  EDFilter::currentContext() const {
    return current_context_;
  }
  
  void
  EDFilter::fillDescription(ParameterSetDescription& iDesc) {
    iDesc.setUnknown();
  }
  
}
  
