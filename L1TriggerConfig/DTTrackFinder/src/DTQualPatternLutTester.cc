//-------------------------------------------------
//
//   Class: DTQualPatternLutTester
//
//   L1 DT Track Finder Quality Pattern Lut Tester
//
//
//
//   Author :
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

#include "L1TriggerConfig/DTTrackFinder/interface/DTQualPatternLutTester.h"

DTQualPatternLutTester::DTQualPatternLutTester(const edm::ParameterSet& ps) : token_{esConsumes()} {}

void DTQualPatternLutTester::analyze(const edm::Event& e, const edm::EventSetup& c) { c.getData(token_).print(); }
