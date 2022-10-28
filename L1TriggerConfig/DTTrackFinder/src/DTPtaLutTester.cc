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

DTPtaLutTester::DTPtaLutTester(const edm::ParameterSet& ps) : token_{esConsumes()} {}

void DTPtaLutTester::analyze(const edm::Event& e, const edm::EventSetup& c) { c.getData(token_).print(); }
