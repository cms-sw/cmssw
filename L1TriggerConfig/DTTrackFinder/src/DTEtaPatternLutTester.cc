//-------------------------------------------------
//
//   Class: DTEtaPatternLutTester
//
//   L1 DT Track Finder Eta Pattern Lut Tester
//
//
//   $Date: 2009/05/04 09:26:10 $
//   $Revision: 1.1 $
//
//   Author :
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

#include "L1TriggerConfig/DTTrackFinder/interface/DTEtaPatternLutTester.h"


DTEtaPatternLutTester::DTEtaPatternLutTester(const edm::ParameterSet& ps) {}


DTEtaPatternLutTester::~DTEtaPatternLutTester() {}


void DTEtaPatternLutTester::analyze(const edm::Event& e, const edm::EventSetup& c) {

  edm::ESHandle<L1MuDTEtaPatternLut> etalut;
  c.get<L1MuDTEtaPatternLutRcd>().get( etalut );
  etalut->print();

}
