//-------------------------------------------------
//
//   Class: DTExtLutTester
//
//   L1 DT Track Finder Extrapolation Lut Tester
//
//
//   $Date: 2008/10/13 03:26:13 $
//   $Revision: 1.4 $
//
//   Author :
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

#include "L1TriggerConfig/DTTrackFinder/interface/DTExtLutTester.h"


DTExtLutTester::DTExtLutTester(const edm::ParameterSet& ps) {}


DTExtLutTester::~DTExtLutTester() {}


void DTExtLutTester::analyze(const edm::Event& e, const edm::EventSetup& c) {

  edm::ESHandle<L1MuDTExtLut> extlut;
  c.get<L1MuDTExtLutRcd>().get( extlut );
  extlut->print();

}
