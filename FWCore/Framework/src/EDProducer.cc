/*----------------------------------------------------------------------
  
$Id: EDProducer.cc,v 1.10 2006/06/20 23:13:27 paterno Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/Framework/src/CPCSentry.h"

namespace edm {
  EDProducer::EDProducer() : ProducerBase(), current_context_(0)  { }

  EDProducer::~EDProducer() { }

  void
  EDProducer::doProduce(Event& e, EventSetup const& c,
			     CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    this->produce(e, c);
  }

  void 
  EDProducer::doBeginJob(EventSetup const& es) {
    this->beginJob(es);
  }
  
  void 
  EDProducer::doEndJob() {
    this->endJob();
  }

  void
  EDProducer::doBeginRun(Run & r, EventSetup const& c,
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    this->beginRun(r, c);
  }

  void
  EDProducer::doEndRun(Run & r, EventSetup const& c,
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    this->endRun(r, c);
  }

  void
  EDProducer::doBeginLuminosityBlock(LuminosityBlock & lb, EventSetup const& c,
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    this->beginLuminosityBlock(lb, c);
  }

  void
  EDProducer::doEndLuminosityBlock(LuminosityBlock & lb, EventSetup const& c,
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    this->endLuminosityBlock(lb, c);
  }


  CurrentProcessingContext const*
  EDProducer::currentContext() const
  {
    return current_context_;
  }
  
}
