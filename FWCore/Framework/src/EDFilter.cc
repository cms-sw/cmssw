/*----------------------------------------------------------------------
  
$Id: EDFilter.cc,v 1.2 2005/07/08 00:09:42 chrjones Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDFilter.h"

namespace edm
{
  EDFilter::~EDFilter()
  { }

  void EDFilter::beginJob( EventSetup const& ) 
  { }
   
  void EDFilter::endJob()
  { }
   
}
  
