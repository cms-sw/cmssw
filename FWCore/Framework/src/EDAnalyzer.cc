/*----------------------------------------------------------------------
  
$Id: EDAnalyzer.cc,v 1.3 2005/07/14 22:50:53 wmtan Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"

namespace edm
{
  EDAnalyzer::~EDAnalyzer()
  { }

  void EDAnalyzer::beginJob(EventSetup const&) 
  { }
   
  void EDAnalyzer::endJob()
  { }
}
  
