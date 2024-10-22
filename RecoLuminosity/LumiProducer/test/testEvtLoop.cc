#ifndef RecoLuminosity_LumiProducer_testEvtLoop_h
#define RecoLuminosity_LumiProducer_testEvtLoop_h
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <iostream>

class testEvtLoop : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
public:
  explicit testEvtLoop(edm::ParameterSet const&);

private:
  void beginJob() override;
  void beginRun(const edm::Run& run, const edm::EventSetup& c) override;
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c) override {}
  void analyze(edm::Event const& e, edm::EventSetup const& c) override;
  void endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void endJob() override;
};  //end class

// -----------------------------------------------------------------

testEvtLoop::testEvtLoop(edm::ParameterSet const& iConfig) {}

// -----------------------------------------------------------------

void testEvtLoop::analyze(edm::Event const& e, edm::EventSetup const&) {
  //std::cout<<"testEvtLoop::analyze"<<std::endl;
}

// -----------------------------------------------------------------
void testEvtLoop::endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c) {
  std::cout << "testEvtLoop::endLuminosityBlock" << std::endl;
  std::cout << "I'm in run " << lumiBlock.run() << " lumi block " << lumiBlock.id().luminosityBlock() << std::endl;
}
// -----------------------------------------------------------------

void testEvtLoop::beginJob() { std::cout << "testEvtLoop::beginJob" << std::endl; }

// -----------------------------------------------------------------

void testEvtLoop::beginRun(const edm::Run& run, const edm::EventSetup& c) {
  std::cout << "testEvtLoop::beginRun" << std::endl;
}

// -----------------------------------------------------------------
void testEvtLoop::endRun(edm::Run const& run, edm::EventSetup const& c) {
  std::cout << "testEvtLoop::endRun" << std::endl;
}

// -----------------------------------------------------------------
void testEvtLoop::endJob() { std::cout << "testEvtLoop::endJob" << std::endl; }

DEFINE_FWK_MODULE(testEvtLoop);
#endif
