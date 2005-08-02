/*----------------------------------------------------------------------
  
$Id: EDProducer.cc,v 1.4 2005/07/21 20:48:17 argiro Exp $

----------------------------------------------------------------------*/

#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/EDProducer.h"

namespace edm
{
  EDProducer::~EDProducer() { }

  void EDProducer::beginJob(EventSetup const&) {
  }

  void EDProducer::endJob() {
  }

  EDProducer::TypeLabelList EDProducer::typeLabelList() const {
    return typeLabelList_;
  }
}
  
