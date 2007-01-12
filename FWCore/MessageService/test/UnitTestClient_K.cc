#include "FWCore/MessageService/test/UnitTestClient_K.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <string>

namespace edmtest
{


void
  UnitTestClient_K::analyze( edm::Event      const & e
                           , edm::EventSetup const & /*unused*/
                              )
{
  
  for (int i=0; i<10; ++i) {
    edm::LogPrint  ("cat_P") << "LogPrint: " << i; 
    edm::LogSystem ("cat_S") << "LogSystem: " << i; 
  }

}  // MessageLoggerClient::analyze()


}  // namespace edmtest


using edmtest::UnitTestClient_K;
DEFINE_FWK_MODULE(UnitTestClient_K);
