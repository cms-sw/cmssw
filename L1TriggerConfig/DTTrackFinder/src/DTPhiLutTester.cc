//-------------------------------------------------
//
//   Class: DTPhiLutTester
//
//   L1 DT Track Finder Phi Assignment Lut Tester
//
//
//   $Date: 2008/10/13 03:26:13 $
//   $Revision: 1.4 $
//
//   Author :
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

#include "L1TriggerConfig/DTTrackFinder/interface/DTPhiLutTester.h"


DTPhiLutTester::DTPhiLutTester(const edm::ParameterSet& ps) {}


DTPhiLutTester::~DTPhiLutTester() {}


void DTPhiLutTester::analyze(const edm::Event& e, const edm::EventSetup& c) {

  edm::ESHandle<L1MuDTPhiLut> philut;
  c.get<L1MuDTPhiLutRcd>().get( philut );
  philut->print();

}
