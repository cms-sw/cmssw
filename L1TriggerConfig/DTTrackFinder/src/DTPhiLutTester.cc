//-------------------------------------------------
//
//   Class: DTPhiLutTester
//
//   L1 DT Track Finder Phi Assignment Lut Tester
//
//
//
//   Author :
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

#include "L1TriggerConfig/DTTrackFinder/interface/DTPhiLutTester.h"

DTPhiLutTester::DTPhiLutTester(const edm::ParameterSet& ps) : token_{esConsumes()} {}

void DTPhiLutTester::analyze(const edm::Event& e, const edm::EventSetup& c) { c.getData(token_).print(); }
