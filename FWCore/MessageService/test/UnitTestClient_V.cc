#include "FWCore/MessageService/test/UnitTestClient_V.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <string>
#include <sstream>

namespace edmtest
{

void
  UTC_V1::analyze( edm::Event      const & e
                            , edm::EventSetup const & /*unused*/
                              )
{
  edm::LogError  ("cat_A")   << "T1 analyze error with identifier " 
  			     << identifier << " event " << ev;
  edm::LogWarning("cat_A")   << "T1 analyze warning with identifier " 
  			     << identifier << " event " << ev;
  edm::LogInfo   ("cat_A")    << "T1 analyze info with identifier " 
  			     << identifier << " event " << ev;
       LogDebug  ("cat_A")    << "T1 analyze debug with identifier " 
  			     << identifier << " event " << ev;
  ev++;
}  

void
  UTC_V1::beginJob(  )
{
  edm::LogWarning("cat_BJ")   << "T1 beginJob warning with identifier " 
  			     << identifier << " event " << ev;
       LogDebug  ("cat_BJ")    << "T1 beginJob debug with identifier " 
  			     << identifier << " event " << ev;
}

void
  UTC_V1::beginRun(edm::Run const& r, edm::EventSetup const&  es)
{
  edm::LogInfo("cat_BR")   << "T1 beginRun info with identifier " 
  			     << identifier << " event " << ev;
       LogDebug  ("cat_BJ")    << "T1 beginRun debug with identifier " 
  			     << identifier << " event " << ev;
}

void
  UTC_V1::beginLuminosityBlock
  		(edm::LuminosityBlock const& lb, edm::EventSetup const& es )
{
  edm::LogWarning("cat_BL")   << "T1 beginLumi warning with identifier " 
  			     << identifier << " event " << ev;
       LogDebug  ("cat_BL")    << "T1 beginLumi debug with identifier " 
  			     << identifier << " event " << ev;
}

void
  UTC_V2::analyze( edm::Event      const & e
                            , edm::EventSetup const & /*unused*/
                              )
{
  edm::LogError  ("cat_A")   << "T1 analyze error with identifier " 
  			     << identifier << " event " << ev;
  edm::LogWarning("cat_A")   << "T1 analyze warning with identifier " 
  			     << identifier << " event " << ev;
  edm::LogInfo  ("cat_A")    << "T1 analyze info with identifier " 
  			     << identifier << " event " << ev;
  ev++;
}  

} // end namespace edmtest

using edmtest::UTC_V1;
using edmtest::UTC_V2;
DEFINE_FWK_MODULE(UTC_V1);
DEFINE_FWK_MODULE(UTC_V2);
