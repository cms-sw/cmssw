// -*- C++ -*-
//
// Package:    TestBeginEndJobAnalyzer
// Class:      TestBeginEndJobAnalyzer
// 
/**\class TestBeginEndJobAnalyzer TestBeginEndJobAnalyzer.cc stubs/TestBeginEndJobAnalyzer/src/TestBeginEndJobAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris Jones
//         Created:  Fri Sep  2 13:54:17 EDT 2005
//
//


#include "FWCore/Framework/test/stubs/TestBeginEndJobAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <memory>
#include <iostream>

void hello(const char * hi) {
  //std::cout << "IN " << hi << std::endl;
}

TestBeginEndJobAnalyzer::TestBeginEndJobAnalyzer(const edm::ParameterSet& /* iConfig */) {
   hello("constr");
}

TestBeginEndJobAnalyzer::~TestBeginEndJobAnalyzer() {
     hello("destr");
  control().destructorCalled = true;
}


void 
TestBeginEndJobAnalyzer::beginJob() {
     hello("bjob");
  control().beginJobCalled = true;
}

void 
TestBeginEndJobAnalyzer::endJob() {
   hello("ejob");
  control().endJobCalled = true;
}

void
TestBeginEndJobAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&) {
  control().beginRunCalled = true;
}

void
TestBeginEndJobAnalyzer::endRun(edm::Run const&, edm::EventSetup const&) {
  control().endRunCalled = true;
}

void
TestBeginEndJobAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {
  control().beginLumiCalled = true;
}

void
TestBeginEndJobAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {
  control().endLumiCalled = true;
}

void
TestBeginEndJobAnalyzer::analyze(const edm::Event& /* iEvent */, const edm::EventSetup& /* iSetup */) {
}
