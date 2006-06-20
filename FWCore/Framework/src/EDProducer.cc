/*----------------------------------------------------------------------
  
$Id: EDProducer.cc,v 1.9 2006/04/20 22:33:22 wmtan Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/Framework/src/CPCSentry.h"

namespace edm {
  EDProducer::EDProducer() : ProducerBase(), current_context_(0)  { }

  EDProducer::~EDProducer() { }

  void EDProducer::doProduce(Event& e, EventSetup const& s,
			     CurrentProcessingContext const* cpc)
  {
    detail::CPCSentry sentry(current_context_, cpc);
    this->produce(e,s);
  }

  void EDProducer::beginJob(EventSetup const&) { }

  void EDProducer::endJob() { }

  CurrentProcessingContext const*
  EDProducer::currentContext() const
  {
    return current_context_;
  }
  
}
