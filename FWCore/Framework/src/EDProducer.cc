/*----------------------------------------------------------------------
  
$Id: EDProducer.cc,v 1.3 2005/07/14 22:50:53 wmtan Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDProducer.h"

namespace edm
{
  EDProducer::~EDProducer()
  { }

  void EDProducer::beginJob( EventSetup const& ) 
  { }

  void EDProducer::endJob()
  { }
  const EDProducer::TypeLabelList& EDProducer::getTypeLabelList() const{
    return productList_;
  }
}
  
