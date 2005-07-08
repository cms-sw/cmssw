/*----------------------------------------------------------------------
  
$Id: EDAnalyzer.cc,v 1.1 2005/05/29 02:29:53 wmtan Exp $

----------------------------------------------------------------------*/

#include "FWCore/CoreFramework/interface/EDAnalyzer.h"

namespace edm
{
  EDAnalyzer::~EDAnalyzer()
  { }

  void EDAnalyzer::beginJob( EventSetup const& ) 
  { }
   
  void EDAnalyzer::endJob()
  { }
}
  
