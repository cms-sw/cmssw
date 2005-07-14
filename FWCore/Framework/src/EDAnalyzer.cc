/*----------------------------------------------------------------------
  
$Id: EDAnalyzer.cc,v 1.2 2005/07/08 00:09:42 chrjones Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"

namespace edm
{
  EDAnalyzer::~EDAnalyzer()
  { }

  void EDAnalyzer::beginJob( EventSetup const& ) 
  { }
   
  void EDAnalyzer::endJob()
  { }
}
  
