//-------------------------------------------------
//
//   Class: DTPtaLutTester
//
//   L1 DT Track Finder Pt Assignment Lut Tester
//
//
//
//   Author :
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

#include "L1TriggerConfig/DTTrackFinder/interface/DTPtaLutTester.h"

DTPtaLutTester::DTPtaLutTester(const edm::ParameterSet& ps) {}

DTPtaLutTester::~DTPtaLutTester() {}

void DTPtaLutTester::analyze(const edm::Event& e, const edm::EventSetup& c) {
  edm::ESHandle<L1MuDTPtaLut> ptalut;
  c.get<L1MuDTPtaLutRcd>().get(ptalut);
  ptalut->print();
}
