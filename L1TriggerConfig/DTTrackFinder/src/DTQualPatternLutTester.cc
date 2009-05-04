//-------------------------------------------------
//
//   Class: DTQualPatternLutTester
//
//   L1 DT Track Finder Quality Pattern Lut Tester
//
//
//   $Date: 2008/10/13 03:26:13 $
//   $Revision: 1.4 $
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
