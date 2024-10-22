#include "L1TriggerConfig/L1ScalesProducers/interface/L1MuScalesTester.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <iomanip>
using std::cout;
using std::endl;

L1MuScalesTester::L1MuScalesTester(const edm::ParameterSet& ps)
    : l1muscalesToken_(esConsumes()), l1muptscaleToken_(esConsumes()), l1gmtscalesToken_(esConsumes()) {}

void L1MuScalesTester::analyze(const edm::Event& e, const edm::EventSetup& es) {
  using namespace edm;

  const char* detnam[] = {"DT", "RPC barrel", "CSC", "RPC forward"};

  L1MuTriggerScales const& l1muscales = es.getData(l1muscalesToken_);

  L1MuTriggerPtScale const& l1muptscale = es.getData(l1muptscaleToken_);

  cout << "**** L1 Mu Pt Scale print *****************************************" << endl;
  printScale(l1muptscale.getPtScale());

  cout << "**** L1 Mu Phi Scale print *****************************************" << endl;
  printScale(l1muscales.getPhiScale());

  cout << "**** L1 Mu GMT eta Scale print *************************************" << endl;
  printScale(l1muscales.getGMTEtaScale());

  for (int i = 0; i < 4; i++) {
    cout << "**** L1 Mu " << detnam[i] << " eta Scale print **************************************" << endl;
    printScale(l1muscales.getRegionalEtaScale(i));
  }

  L1MuGMTScales const& l1gmtscales = es.getData(l1gmtscalesToken_);

  for (int i = 0; i < 4; i++) {
    cout << "**** L1 GMT " << detnam[i] << " reduced eta Scale print **************************************" << endl;
    printScale(l1gmtscales.getReducedEtaScale(i));
  }

  cout << "**** L1 GMT delta eta Scale print *************************************" << endl;
  printScale(l1gmtscales.getDeltaEtaScale(0));

  cout << "**** L1 GMT delta phi Scale print *************************************" << endl;
  printScale(l1gmtscales.getDeltaPhiScale());

  for (int i = 0; i < 4; i++) {
    cout << "**** L1 GMT " << detnam[i] << " overlap eta Scale print **************************************" << endl;
    printScale(l1gmtscales.getOvlEtaScale(i));
  }

  //   cout << "**** L1 GMT calo eta Scale print *************************************" << endl;
  //   printScale(l1gmtscales.getCaloEtaScale());
}

void L1MuScalesTester::printScale(const L1MuScale* l1muscale) { cout << l1muscale->print(); }
