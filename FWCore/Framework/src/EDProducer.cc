/*----------------------------------------------------------------------
  
$Id: EDProducer.cc,v 1.5 2005/08/02 22:23:18 wmtan Exp $

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

   boost::function<void(const BranchDescription&)> EDProducer::registrationCallback() const {
      return callWhenNewProductsRegistered_;
   }

}
  
