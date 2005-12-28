/*----------------------------------------------------------------------
  
$Id: EDProducer.cc,v 1.7 2005/12/28 00:32:04 wmtan Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDProducer.h"

namespace edm {
  EDProducer::~EDProducer() { }

  void EDProducer::beginJob(EventSetup const&) {
  }

  void EDProducer::endJob() {
  }

   boost::function<void(const BranchDescription&)> EDProducer::registrationCallback() const {
      return callWhenNewProductsRegistered_;
   }
}
