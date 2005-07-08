/*----------------------------------------------------------------------
  
$Id: EDProducer.cc,v 1.1 2005/05/29 02:29:53 wmtan Exp $

----------------------------------------------------------------------*/

#include "FWCore/CoreFramework/interface/EDProducer.h"

namespace edm
{
  EDProducer::~EDProducer()
  { }

  void EDProducer::beginJob( EventSetup const& ) 
  { }

  void EDProducer::endJob()
  { }
}
  
