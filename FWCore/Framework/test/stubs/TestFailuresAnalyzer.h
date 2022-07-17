#ifndef Framework_TestFailuresAnalyzer_h
#define Framework_TestFailuresAnalyzer_h
// -*- C++ -*-
//
// Package:     test
// Class  :     TestFailuresAnalyzer
//
/**\class TestFailuresAnalyzer TestFailuresAnalyzer.h Framework/test/interface/TestFailuresAnalyzer.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Sep  2 14:17:17 EDT 2005
//

// system include files
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
// user include files

// forward declarations
class TestFailuresAnalyzer : public edm::global::EDAnalyzer<> {
public:
  explicit TestFailuresAnalyzer(const edm::ParameterSet&);

  void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const final;

  void beginJob() final;
  void endJob() final;

private:
  // ----------member data ---------------------------
  const int whichFailure_;
  const unsigned long long eventToThrow_;
};

#endif /* test_TestFailuresAnalyzer_h */
