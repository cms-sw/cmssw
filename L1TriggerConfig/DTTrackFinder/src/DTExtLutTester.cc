//-------------------------------------------------
//
//   Class: DTExtLutTester
//
//   L1 DT Track Finder Extrapolation Lut Tester
//
//
//
//   Author :
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

#include "L1TriggerConfig/DTTrackFinder/interface/DTExtLutTester.h"

DTExtLutTester::DTExtLutTester(const edm::ParameterSet& ps) : token_{esConsumes()} {}

void DTExtLutTester::analyze(const edm::Event& e, const edm::EventSetup& c) { c.getData(token_).print(); }
