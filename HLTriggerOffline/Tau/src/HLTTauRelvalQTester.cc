#include "HLTriggerOffline/Tau/interface/HLTTauRelvalQTester.h"
//#include "DataFormats/Math/interface/LorentzVector.h"

HLTTauRelvalQTester::HLTTauRelvalQTester(const edm::ParameterSet &ps) : QualityTester(ps) {
  refMothers_ = consumes<std::vector<int>>(ps.getParameter<edm::InputTag>("refMothers"));
  mothers_ = ps.getParameter<std::vector<int>>("mothers");
  runQTests = false;
}

HLTTauRelvalQTester::~HLTTauRelvalQTester() {}

void HLTTauRelvalQTester::analyze(const edm::Event &e, const edm::EventSetup &c) {
  edm::Handle<std::vector<int>> refMothers;
  if (e.getByToken(refMothers_, refMothers))
    for (unsigned int i = 0; i < refMothers->size(); ++i) {
      int mother = (*refMothers)[i];
      for (unsigned int j = 0; j < mothers_.size(); ++j) {
        if (mothers_[j] == mother)
          runQTests = true;
      }

      if (runQTests) {
        QualityTester::analyze(e, c);
      }
    }
}

void HLTTauRelvalQTester::endLuminosityBlock(edm::LuminosityBlock const &lumiSeg, edm::EventSetup const &c) {
  QualityTester::endLuminosityBlock(lumiSeg, c);
}

void HLTTauRelvalQTester::endRun(const edm::Run &r, const edm::EventSetup &c) { QualityTester::endRun(r, c); }

void HLTTauRelvalQTester::endJob() { QualityTester::endJob(); }
