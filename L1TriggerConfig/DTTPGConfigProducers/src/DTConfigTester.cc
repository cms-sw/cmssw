#include "L1TriggerConfig/DTTPGConfigProducers/src/DTConfigTester.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManager.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManagerRcd.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTBtiId.h"
#include "DataFormats/MuonDetId/interface/DTTracoId.h"
#include "DataFormats/MuonDetId/interface/DTSectCollId.h"

using std::cout;
using std::endl;

DTConfigTester::DTConfigTester(const edm::ParameterSet& ps) {
	cout << "DTConfigTester::DTConfigTester" << endl;
}

DTConfigTester::~DTConfigTester() {

}

void DTConfigTester::analyze(const edm::Event& e, const edm::EventSetup& es) {
   cout << "DTConfigTester::analyze" << endl;
   cout <<" I AM IN RUN NUMBER "<<e.id().run() <<endl;
   cout <<" ---EVENT NUMBER "<<e.id().event() <<endl;

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
   dtConfig->getDTConfigLUTs(chid)->print();
   dtConfig->getDTConfigSectColl(scid)->print();
   
}
