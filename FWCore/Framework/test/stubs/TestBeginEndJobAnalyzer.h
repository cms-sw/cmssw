#ifndef Framework_TestBeginEndJobAnalyzer_h
#define Framework_TestBeginEndJobAnalyzer_h
// -*- C++ -*-
//
// Package:     test
// Class  :     TestBeginEndJobAnalyzer
//
/**\class TestBeginEndJobAnalyzer TestBeginEndJobAnalyzer.h Framework/test/interface/TestBeginEndJobAnalyzer.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Sep  2 14:17:17 EDT 2005
//

// system include files
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
// user include files

// forward declarations
class TestBeginEndJobAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
public:
  explicit TestBeginEndJobAnalyzer(const edm::ParameterSet&);
  ~TestBeginEndJobAnalyzer();

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  void beginJob() override;
  void endJob() override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  struct Control {
    bool beginJobCalled = false;
    bool endJobCalled = false;
    bool beginRunCalled = false;
    bool endRunCalled = false;
    bool beginLumiCalled = false;
    bool endLumiCalled = false;
    bool destructorCalled = false;
  };

  static Control& control();

private:
  // ----------member data ---------------------------
};

#endif /* test_TestBeginEndJobAnalyzer_h */
