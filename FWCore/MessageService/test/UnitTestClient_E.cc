#include "FWCore/MessageService/test/UnitTestClient_E.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <string>

namespace edmtest
{


void
  UnitTestClient_E::analyze( edm::Event      const & /*unused*/
                           , edm::EventSetup const & /*unused*/
                              )
{
    edm::LogInfo   ("expect_overall_unnamed")   
    << "The following outputs are expected: \n"  
    << "unlisted_category    appearing in events 1,2,3,4,5,10,15,25,45 \n";
    
    edm::LogInfo   ("expect_overall_specific")   
    << "The following outputs are expected: \n"  
    << "lim3bydefault    appearing in events 1,2,3,6,9,15,27 \n";
 
    edm::LogInfo   ("expect_supercede_specific")   
    << "The following outputs are expected: \n"  
    << "lim2bycommondefault appearing in events 1,2,3,4,5,6,7,8,16,24,40 \n";
 
    edm::LogInfo   ("expect_non_supercede_common_specific")   
    << "The following outputs are expected: \n"  
    << "lim2bycommondefault appearing in events 1,2,4,6,10,18,34 \n";

    edm::LogInfo   ("expect_specific")   
    << "The following outputs are expected: \n"  
    << "lim0bydefaults appearing in events 1,2,3,4,5,6,12,18,30 \n";
 
  for (int i=1; i<=50; ++i) {
    edm::LogInfo   ("unlisted_category")   << 
  	"message with overall default limit of 5: " << i;
    edm::LogInfo   ("lim3bydefault")   << 
  	"message with specific overall default limit of 3: " << i;
    edm::LogInfo   ("lim2bycommondefault")   << 
  	"message with specific overall default limit of 2: " << i;
    edm::LogInfo   ("lim0bydefaults")   << 
  	"message with overall default and dest default limit of 0: " << i;
//    edm::LogInfo   ("lim3bydefault")   << 
//  	"message with overall default limit (superceded) of 2: " << i;
   }
   
}  // MessageLoggerClient::analyze()


}  // namespace edmtest


using edmtest::UnitTestClient_E;
DEFINE_FWK_MODULE(UnitTestClient_E);
