#include "L1TriggerConfig/DTTPGConfigProducers/src/DTConfigTester.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManager.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManagerRcd.h"

#include "DataFormats/MuonDetId/interface/DTBtiId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSectCollId.h"
#include "DataFormats/MuonDetId/interface/DTTracoId.h"

using std::cout;
using std::endl;

DTConfigTester::DTConfigTester(const edm::ParameterSet &ps) {
  cout << "DTConfigTester::DTConfigTester()" << endl;

  my_wh = ps.getUntrackedParameter<int>("wheel");
  my_sec = ps.getUntrackedParameter<int>("sector");
  my_st = ps.getUntrackedParameter<int>("station");
  my_traco = ps.getUntrackedParameter<int>("traco");
  my_bti = ps.getUntrackedParameter<int>("bti");
  my_sl = ps.getUntrackedParameter<int>("sl");
  my_configToken = esConsumes();
}

void DTConfigTester::analyze(const edm::Event &e, const edm::EventSetup &es) {
  cout << "DTConfigTester::analyze()" << endl;
  cout << "\tRun number :" << e.id().run() << endl;
  cout << "\tEvent number :" << e.id().event() << endl;

  using namespace edm;

  ESHandle<DTConfigManager> dtConfig = es.getHandle(my_configToken);

  cout << "\tPrint configuration :" << endl;

  DTBtiId btiid(my_wh, my_st, my_sec, my_sl, my_bti);
  DTTracoId tracoid(my_wh, my_st, my_sec, my_traco);
  DTChamberId chid(my_wh, my_st, my_sec);
  DTSectCollId scid(my_wh, my_sec);

  dtConfig->getDTConfigBti(btiid)->print();
  dtConfig->getDTConfigTraco(tracoid)->print();
  dtConfig->getDTConfigTSTheta(chid)->print();
  dtConfig->getDTConfigTSPhi(chid)->print();
  dtConfig->getDTConfigTrigUnit(chid)->print();

  if (dtConfig->lutFromDB())
    dtConfig->getDTConfigLUTs(chid)->print();
  else {
    cout << "******************************************************************"
            "*************"
         << endl;
    cout << "*              DTTrigger configuration : LUT parameters from "
            "GEOMETRY         *"
         << endl;
    cout << "******************************************************************"
            "*************"
         << endl;
  }

  dtConfig->getDTConfigSectColl(scid)->print();
  dtConfig->getDTConfigPedestals()->print();

  /*
     // 100209 SV testing luts for each chamber type: keep in case lut from DB
     debug is necessary DTChamberId chid1(-2,3,1); cout << "\n CHAMBER -2 3 1"
     << endl; dtConfig->getDTConfigLUTs(chid1)->print();

     DTChamberId chid2(0,2,2);
     cout << "\n CHAMBER 0 2 2" << endl;
     dtConfig->getDTConfigLUTs(chid2)->print();

     DTChamberId chid3(-2,4,8);
     cout << "\n CHAMBER -2 4 8" << endl;
     dtConfig->getDTConfigLUTs(chid3)->print();

     DTChamberId chid4(1,4,12);
     cout << "\n CHAMBER 1 4 12" << endl;
     dtConfig->getDTConfigLUTs(chid4)->print();

     DTChamberId chid5(-2,4,5);
     cout << "\n CHAMBER -2 4 5 " << endl;
     dtConfig->getDTConfigLUTs(chid5)->print();

     DTChamberId chid6(0,4,2);
     cout << "\n CHAMBER 0 4 2" << endl;
     dtConfig->getDTConfigLUTs(chid6)->print();

     DTChamberId chid7(-2,4,9);
     cout << "\n CHAMBER -2 4 9" << endl;
     dtConfig->getDTConfigLUTs(chid7)->print();

     DTChamberId chid8(0,4,11);
     cout << "\n CHAMBER 0 4 11" << endl;
     dtConfig->getDTConfigLUTs(chid8)->print();

     DTChamberId chid9(-2,1,1);
     cout << "\n CHAMBER -2 1 1" << endl;
     dtConfig->getDTConfigLUTs(chid9)->print();

     DTChamberId chid10(-2,4,13);
     cout << "\n CHAMBER -2 4 13" << endl;
     dtConfig->getDTConfigLUTs(chid10)->print();

     DTChamberId chid11(1,4,4);
     cout << "\n CHAMBER 1 4 4 " << endl;
     dtConfig->getDTConfigLUTs(chid11)->print();

     DTChamberId chid12(-2,4,14);
     cout << "\n CHAMBER -2 4 14" << endl;
     dtConfig->getDTConfigLUTs(chid12)->print();

     DTChamberId chid13(0,4,10);
     cout << "\n CHAMBER 0 4 10" << endl;
     dtConfig->getDTConfigLUTs(chid13)->print();

     DTChamberId chid14(-2,4,11);
     cout << "\n CHAMBER -2 4 11" << endl;
     dtConfig->getDTConfigLUTs(chid14)->print();

     DTChamberId chid15(1,4,9);
     cout << "\n CHAMBER 1 4 9" << endl;
     dtConfig->getDTConfigLUTs(chid15)->print();

     DTChamberId chid16(0,1,2);
     cout << "\n CHAMBER 0 1 2 " << endl;
     dtConfig->getDTConfigLUTs(chid16)->print();

     DTChamberId chid17(-2,2,1);
     cout << "\n CHAMBER -2 2 1" << endl;
     dtConfig->getDTConfigLUTs(chid17)->print();

     DTChamberId chid18(0,3,2);
     cout << "\n CHAMBER 0 3 2 " << endl;
     dtConfig->getDTConfigLUTs(chid18)->print();

     DTChamberId chid19(-2,4,10);
     cout << "\n CHAMBER 0 2 2" << endl;
     dtConfig->getDTConfigLUTs(chid19)->print();

     DTChamberId chid20(0,4,14);
     cout << "\n CHAMBER 0 4 14" << endl;
     dtConfig->getDTConfigLUTs(chid20)->print();

     DTChamberId chid21(-2,4,12);
     cout << "\n CHAMBER -2 4 12" << endl;
     dtConfig->getDTConfigLUTs(chid21)->print();

     DTChamberId chid22(1,4,8);
     cout << "\n CHAMBER 1 4 8" << endl;
     dtConfig->getDTConfigLUTs(chid22)->print();

     DTChamberId chid23(-2,4,1);
     cout << "\n CHAMBER -2 4 1" << endl;
     dtConfig->getDTConfigLUTs(chid23)->print();

     DTChamberId chid24(0,4,6);
     cout << "\n CHAMBER 0 4 6" << endl;
     dtConfig->getDTConfigLUTs(chid24)->print();

     DTChamberId chid25(-2,4,4);
     cout << "\n CHAMBER -2 4 4" << endl;
     dtConfig->getDTConfigLUTs(chid25)->print();

     DTChamberId chid26(1,4,13);
     cout << "\n CHAMBER 1 4 13 " << endl;
     dtConfig->getDTConfigLUTs(chid26)->print();

  */
  return;
}
