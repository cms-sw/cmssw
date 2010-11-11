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

  cout << "DTConfigTester::DTConfigTester()" << endl;

  my_wh    = ps.getUntrackedParameter<int>("wheel");
  my_sec   = ps.getUntrackedParameter<int>("sector");
  my_st    = ps.getUntrackedParameter<int>("station");
  my_traco = ps.getUntrackedParameter<int>("traco");
  my_bti   = ps.getUntrackedParameter<int>("bti");

}

DTConfigTester::~DTConfigTester() {

  cout << "DTConfigTester::~DTConfigTester()" << endl;

}

void DTConfigTester::analyze(const edm::Event& e, const edm::EventSetup& es) {

   cout << "DTConfigTester::analyze()" << endl;
   cout << "\tRun number :" << e.id().run() << endl;
   cout << "\tEvent number :" << e.id().event() << endl;

   using namespace edm;

   ESHandle< DTConfigManager > dtConfig ;
   es.get< DTConfigManagerRcd >().get( dtConfig ) ;

   cout << "\tPrint configuration :" << endl;

   DTBtiId btiid(my_wh,my_st,my_sec,my_traco,my_bti);
   DTTracoId tracoid(my_wh,my_st,my_sec,my_traco);
   DTChamberId chid(my_wh,my_st,my_sec);
   DTSectCollId scid(my_wh,my_sec);

   dtConfig->getDTConfigBti(btiid)->print();
   dtConfig->getDTConfigTraco(tracoid)->print();
   dtConfig->getDTConfigTSTheta(chid)->print();
   dtConfig->getDTConfigTSPhi(chid)->print();
   dtConfig->getDTConfigTrigUnit(chid)->print();
   dtConfig->getDTConfigLUTs(chid)->print();
   dtConfig->getDTConfigSectColl(scid)->print();
    dtConfig->getDTConfigPedestals()->print();
//    dtConfig->getDTConfigPedestals()->print();
   
}
