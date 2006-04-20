/*----------------------------------------------------------------------
  
$Id: EDProducer.cc,v 1.8 2005/12/28 00:48:40 wmtan Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDProducer.h"

namespace edm {
  EDProducer::EDProducer() : ProducerBase() {
  }

  EDProducer::~EDProducer() {
  }

  void EDProducer::beginJob(EventSetup const&) {
  }

  void EDProducer::endJob() {
  }
}
