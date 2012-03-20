#include "FWCore/MessageService/test/UnitTestClient_F.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <string>

namespace edmtest
{


void
  UnitTestClient_F::analyze( edm::Event      const & /*unused*/
                           , edm::EventSetup const & /*unused*/
                              )
{
    edm::LogInfo   ("expect_overall_unnamed")   
    << "The following outputs are expected: \n"  
    << "unlisted_category    appearing in events 1,6,11,16,21,26,31,36,41,46 \n";
    
    edm::LogInfo   ("expect_overall_specific")   
    << "The following outputs are expected: \n"  
    << "int7bycommondefault    appearing in events 1,8,15,22,29,36,43,50 \n";
 
    edm::LogInfo   ("expect_supercede_specific")   
    << "The following outputs are expected: \n"  
    << "int7bycommondefault appearing in events 1,11,21,31,41 \n";
 
    edm::LogInfo   ("expect_non_supercede_common_specific")   
    << "The following outputs are expected: \n"  
    << "int7bycommondefault appearing in events 1,19,37 \n"
    << "unlisted_category appearing in events 1,27 \n";

    edm::LogInfo   ("expect_specific")   
    << "The following outputs are expected: \n"  
    << "int25bydefaults appearing in events 1,13,25,37,49 \n"
    << "unlisted_category appearing in events 1,31 \n";
 
  for (int i=1; i<=50; ++i) {
    edm::LogInfo   ("unlisted_category")   << 
  	"message with overall default interval of 5: " << i;
    edm::LogInfo   ("int4bydefault")   << 
  	"message with specific overall default interval of 4: " << i;
    edm::LogInfo   ("int7bycommondefault")   << 
  	"message with specific overall default interval of 7: " << i;
    edm::LogInfo   ("int25bydefaults")   << 
  	"message with overall default and dest default interval of 25: " << i;
   }
   
}  // MessageLoggerClient::analyze()


}  // namespace edmtest


using edmtest::UnitTestClient_F;
DEFINE_FWK_MODULE(UnitTestClient_F);
