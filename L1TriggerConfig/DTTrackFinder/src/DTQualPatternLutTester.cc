//-------------------------------------------------
//
//   Class: DTQualPatternLutTester
//
//   L1 DT Track Finder Quality Pattern Lut Tester
//
//
//   $Date: 2009/05/04 09:26:10 $
//   $Revision: 1.1 $
//
//   Author :
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

#include "L1TriggerConfig/DTTrackFinder/interface/DTQualPatternLutTester.h"


DTQualPatternLutTester::DTQualPatternLutTester(const edm::ParameterSet& ps) {}


DTQualPatternLutTester::~DTQualPatternLutTester() {}


void DTQualPatternLutTester::analyze(const edm::Event& e, const edm::EventSetup& c) {

  edm::ESHandle<L1MuDTQualPatternLut> qualut;
  c.get<L1MuDTQualPatternLutRcd>().get( qualut );
  qualut->print();

}
