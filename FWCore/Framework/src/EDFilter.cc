/*----------------------------------------------------------------------
  
$Id: EDFilter.cc,v 1.1 2005/05/29 02:29:53 wmtan Exp $

----------------------------------------------------------------------*/

#include "FWCore/CoreFramework/interface/EDFilter.h"

namespace edm
{
  EDFilter::~EDFilter()
  { }

  void EDFilter::beginJob( EventSetup const& ) 
  { }
   
  void EDFilter::endJob()
  { }
   
}
  
