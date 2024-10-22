//-------------------------------------------------
//
//   Class: DTEtaPatternLutTester
//
//   L1 DT Track Finder Eta Pattern Lut Tester
//
//
//
//   Author :
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

#include "L1TriggerConfig/DTTrackFinder/interface/DTEtaPatternLutTester.h"

DTEtaPatternLutTester::DTEtaPatternLutTester(const edm::ParameterSet& ps) : token_{esConsumes()} {}

void DTEtaPatternLutTester::analyze(const edm::Event& e, const edm::EventSetup& c) { c.getData(token_).print(); }
