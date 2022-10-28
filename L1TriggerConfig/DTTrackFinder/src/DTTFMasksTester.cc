//-------------------------------------------------
//
//   Class: DTTFMasksTester
//
//   L1 DT Track Finder Parameters Tester
//
//
//
//   Author :
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

#include "L1TriggerConfig/DTTrackFinder/interface/DTTFMasksTester.h"

DTTFMasksTester::DTTFMasksTester(const edm::ParameterSet& ps) : token_{esConsumes()} {}

void DTTFMasksTester::analyze(const edm::Event& e, const edm::EventSetup& c) { c.getData(token_).print(); }
