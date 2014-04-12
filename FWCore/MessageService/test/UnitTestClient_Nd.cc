#include "FWCore/MessageService/test/UnitTestClient_Nd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <string>

namespace edmtest
{


void
  UnitTestClient_Nd::analyze( edm::Event      const & /*unused*/
                           , edm::EventSetup const & /*unused*/
                              )
{
  std::string empty_;
  std::string file_ = "nameOfFile";
       LogDebug  ("ridiculously_long_category_name_to_make_header_wrap_A")   
       		<< "LogDebug was used to send this message";
       LogDebug  ("ridiculously_long_category_name_to_make_header_wrap_B")  
        	<< "LogDebug was used to send this other message";
  edm::LogInfo   ("ridiculously_long_category_name_to_make_header_wrap_A")  
  		<< "LogInfo was used to send this message";
  edm::LogInfo   ("ridiculously_long_category_name_to_make_header_wrap_B")   
  		<< "LogInfo was used to send this other message";

 }  // MessageLoggerClient::analyze()


}  // namespace edmtest


using edmtest::UnitTestClient_Nd;
DEFINE_FWK_MODULE(UnitTestClient_Nd);
