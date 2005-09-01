/*----------------------------------------------------------------------
  
$Id: EDFilter.cc,v 1.3 2005/07/14 22:50:53 wmtan Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDFilter.h"

namespace edm
{
  EDFilter::~EDFilter()
  { }

  void EDFilter::beginJob(EventSetup const&) 
  { }
   
  void EDFilter::endJob()
  { }
   
}
  
