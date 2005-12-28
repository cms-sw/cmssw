/*----------------------------------------------------------------------
  
$Id: EDProducer.cc,v 1.6 2005/10/11 19:32:28 chrjones Exp $

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

   boost::function<void(const BranchDescription&)> EDProducer::registrationCallback() const {
      return callWhenNewProductsRegistered_;
   }

}
  
