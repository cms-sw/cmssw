#include "L1TriggerConfig/L1ScalesProducers/interface/L1ScalesTester.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"
#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloEcalScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"

#include <iostream>

using std::cout;
using std::endl;

L1ScalesTester::L1ScalesTester(const edm::ParameterSet& ps) { cout << "Constructing a L1ScalesTester" << endl; }

L1ScalesTester::~L1ScalesTester() {}

void L1ScalesTester::analyze(const edm::Event& e, const edm::EventSetup& es) {
  using namespace edm;

  ESHandle<L1CaloEtScale> emScale;
  es.get<L1EmEtScaleRcd>().get(emScale);

  cout << "L1EmEtScaleRcd :" << endl;
  emScale->print(cout);
  cout << endl;

  ESHandle<L1CaloEcalScale> ecalScale;
  es.get<L1CaloEcalScaleRcd>().get(ecalScale);

  ESHandle<L1CaloHcalScale> hcalScale;
  es.get<L1CaloHcalScaleRcd>().get(hcalScale);

  cout << " L1ColoEcalScale  :" << endl;
  ecalScale->print(cout);
  cout << endl;

  cout << " L1ColoHcalScale  :" << endl;
  hcalScale->print(cout);
  cout << endl;

  ESHandle<L1CaloEtScale> jetScale;
  es.get<L1JetEtScaleRcd>().get(jetScale);

  cout << "L1JetEtScaleRcd :" << endl;
  jetScale->print(cout);
  cout << endl;

  // test EM lin-rank conversion
  cout << "Testing EM linear-to-rank conversion" << endl;
  for (unsigned short i = 0; i < 32; i++) {
    unsigned rank = emScale->rank(i);
    cout << "EM linear : " << i << ", Et : " << i * emScale->linearLsb() << " GeV, rank : " << rank << endl;
  }
  cout << endl;

  // test jet lin-rank conversion
  cout << "Testing jet linear-to-rank conversion" << endl;
  for (unsigned short i = 0; i < 32; i++) {
    unsigned rank = jetScale->rank(i);
    cout << "jet linear : " << i << ", Et : " << i * jetScale->linearLsb() << " GeV, rank : " << rank << endl;
  }
  cout << endl;

  // test EM rank-et conversion
  cout << "Testing EM rank-to-Et conversion" << endl;
  for (unsigned i = 0; i < 32; i++) {
    double et = emScale->et(i);
    cout << "EM rank : " << i << " Et : " << et << " GeV" << endl;
  }
  cout << endl;

  // test jet rank-et conversion
  cout << "Testing jet rank-to-Et conversion" << endl;
  for (unsigned i = 0; i < 32; i++) {
    double et = jetScale->et(i);
    cout << "jet rank : " << i << " Et : " << et << " GeV" << endl;
  }
  cout << endl;
}
