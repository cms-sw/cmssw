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
// $Id: TestBeginEndJobAnalyzer.cc,v 1.7 2008/04/04 16:11:04 wdd Exp $
//
//


#include "FWCore/Framework/test/stubs/TestBeginEndJobAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <memory>

TestBeginEndJobAnalyzer::TestBeginEndJobAnalyzer(const edm::ParameterSet& /* iConfig */) {
}

TestBeginEndJobAnalyzer::~TestBeginEndJobAnalyzer() {
  destructorCalled = true;
}

bool TestBeginEndJobAnalyzer::beginJobCalled = false;
bool TestBeginEndJobAnalyzer::endJobCalled = false;
bool TestBeginEndJobAnalyzer::beginRunCalled = false;
bool TestBeginEndJobAnalyzer::endRunCalled = false;
bool TestBeginEndJobAnalyzer::beginLumiCalled = false;
bool TestBeginEndJobAnalyzer::endLumiCalled = false;
bool TestBeginEndJobAnalyzer::destructorCalled = false;

void 
TestBeginEndJobAnalyzer::beginJob() {
  beginJobCalled = true;
}

void 
TestBeginEndJobAnalyzer::endJob() {
  endJobCalled = true;
}

void
TestBeginEndJobAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&) {
  beginRunCalled = true;
}

void
TestBeginEndJobAnalyzer::endRun(edm::Run const&, edm::EventSetup const&) {
  endRunCalled = true;
}

void
TestBeginEndJobAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {
  beginLumiCalled = true;
}

void
TestBeginEndJobAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {
  endLumiCalled = true;
}

void
TestBeginEndJobAnalyzer::analyze(const edm::Event& /* iEvent */, const edm::EventSetup& /* iSetup */) {
}
