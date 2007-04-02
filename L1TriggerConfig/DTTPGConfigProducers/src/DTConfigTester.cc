#include "L1TriggerConfig/DTTPGConfigProducers/src/DTConfigTester.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManager.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManagerRcd.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "L1Trigger/DTUtilities/interface/DTBtiId.h"
#include "L1Trigger/DTUtilities/interface/DTTracoId.h"
#include "L1Trigger/DTUtilities/interface/DTSectCollId.h"

using std::cout;
using std::endl;

DTConfigTester::DTConfigTester(const edm::ParameterSet& ps) {
  cout << "Constructing a DTConfigTester" << endl;
}

DTConfigTester::~DTConfigTester() {

}

void DTConfigTester::analyze(const edm::Event& e, const edm::EventSetup& es) {
   using namespace edm;

   ESHandle< DTConfigManager > dtConfig ;
   es.get< DTConfigManagerRcd >().get( dtConfig ) ;

   cout << "DTConfigManagerRcd : Print some Config stuff" << endl;
   DTBtiId btiid(1,1,1,1,1);
   DTTracoId tracoid(1,1,1,1);
   DTChamberId chid(1,1,1);
   DTSectCollId scid(1,1);
   cout <<"BtiMap & TracoMap Size for chamber (1,1,1):" << dtConfig->getDTConfigBtiMap(chid).size() << " " << dtConfig->getDTConfigTracoMap(chid).size() << endl; 

   dtConfig->getDTConfigBti(btiid)->print();
   dtConfig->getDTConfigTraco(tracoid)->print();
   dtConfig->getDTConfigTSTheta(chid)->print();
   dtConfig->getDTConfigTSPhi(chid)->print();
   dtConfig->getDTConfigSectColl(scid)->print();
   

}
