//-------------------------------------------------
//
//   Class: DTTFParametersTester
//
//   L1 DT Track Finder Parameters Tester
//
//
//
//   Author :
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

#include "L1TriggerConfig/DTTrackFinder/interface/DTTFParametersTester.h"

DTTFParametersTester::DTTFParametersTester(const edm::ParameterSet& ps) : token_{esConsumes()} {}

void DTTFParametersTester::analyze(const edm::Event& e, const edm::EventSetup& c) { c.getData(token_).print(); }
