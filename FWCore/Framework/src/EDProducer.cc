/*----------------------------------------------------------------------
  
$Id: EDProducer.cc,v 1.2 2005/07/08 00:09:42 chrjones Exp $

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
}
  
