#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/PatCandidates/interface/EventHypothesis.h"
#include "DataFormats/PatCandidates/interface/EventHypothesisLooper.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/MuonReco/interface/Muon.h"

class TestEventHypothesisReader : public edm::stream::EDAnalyzer<> {
public:
  TestEventHypothesisReader(const edm::ParameterSet &iConfig);
  void analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) override;
  void runTests(const pat::EventHypothesis &h);

private:
  edm::EDGetTokenT<std::vector<pat::EventHypothesis> > hypsToken_;
  edm::EDGetTokenT<edm::ValueMap<double> > deltaRsHToken_;
};

TestEventHypothesisReader::TestEventHypothesisReader(const edm::ParameterSet &iConfig)
    : hypsToken_(consumes<std::vector<pat::EventHypothesis> >(iConfig.getParameter<edm::InputTag>("events"))),
      deltaRsHToken_(consumes<edm::ValueMap<double> >(
          edm::InputTag((iConfig.getParameter<edm::InputTag>("events")).label(), "deltaR"))) {}

void TestEventHypothesisReader::runTests(const pat::EventHypothesis &h) {
  using namespace std;
  using namespace pat::eventhypothesis;
  cout << "Test 1: Print the muon " << h["mu"]->pt() << endl;

  for (size_t i = 0; i < h.count() - 2; ++i) {
    cout << "Test 2." << (i + 1) << ": Getting of the other jets: " << h.get("other jet", i)->et() << endl;
  }

  cout << "Test 3: count: " << (h.count() - 2) << " vs " << h.count("other jet") << endl;

  cout << "Test 4: regexp count: " << (h.count() - 1) << " vs " << h.count(".*jet") << endl;

  cout << "Test 5.0: all with muon: " << h.all("mu").size() << endl;
  cout << "Test 5.1: all with muon: " << h.all("mu").front()->pt() << endl;
  cout << "Test 5.2: all with other jets: " << h.all("other jet").size() << endl;
  cout << "Test 5.3: all with regex: " << h.all(".*jet").size() << endl;

  cout << "Test 6.0: get as : " << h.getAs<reco::CaloJet>("nearest jet")->maxEInHadTowers() << endl;

  cout << "Loopers" << endl;
  cout << "Test 7.0: simple looper on all" << endl;
  for (CandLooper jet = h.loop(); jet; ++jet) {
    cout << "\titem " << jet.index() << ", role " << jet.role() << ": " << jet->et() << endl;
  }
  cout << "Test 7.1: simple looper on jets" << endl;
  for (CandLooper jet = h.loop(".*jet"); jet; ++jet) {
    cout << "\titem " << jet.index() << ", role " << jet.role() << ": " << jet->et() << endl;
  }
  cout << "Test 7.2: loopAs on jets" << endl;
  for (Looper<reco::CaloJet> jet = h.loopAs<reco::CaloJet>(".*jet"); jet; ++jet) {
    cout << "\titem " << jet.index() << ", role " << jet.role() << ": " << jet->maxEInHadTowers() << endl;
  }
}

void TestEventHypothesisReader::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;
  using namespace std;
  using reco::Candidate;
  using reco::CandidatePtr;

  Handle<vector<pat::EventHypothesis> > hyps;
  iEvent.getByToken(hypsToken_, hyps);

  Handle<ValueMap<double> > deltaRsH;
  iEvent.getByToken(deltaRsHToken_, deltaRsH);
  const ValueMap<double> &deltaRs = *deltaRsH;

  for (size_t i = 0, n = hyps->size(); i < n; ++i) {
    const pat::EventHypothesis &h = (*hyps)[i];

    std::cout << "Hypothesis " << (i + 1) << ": " << std::endl;
    CandidatePtr mu = h["mu"];
    std::cout << "   muon : pt = " << mu->pt() << ", eta = " << mu->eta() << ", phi = " << mu->phi() << std::endl;
    CandidatePtr jet = h["nearest jet"];
    std::cout << "   n jet: pt = " << jet->pt() << ", eta = " << jet->eta() << ", phi = " << jet->phi() << std::endl;

    for (pat::EventHypothesis::CandLooper j2 = h.loop("other jet"); j2; ++j2) {
      std::cout << "   0 jet: pt = " << j2->pt() << ", eta = " << j2->eta() << ", phi = " << j2->phi() << std::endl;
    }

    Ref<vector<pat::EventHypothesis> > key(hyps, i);
    std::cout << "   deltaR: " << deltaRs[key] << "\n" << std::endl;

    runTests(h);
  }

  std::cout << "Found " << hyps->size() << " possible options"
            << "\n\n"
            << std::endl;
}

DEFINE_FWK_MODULE(TestEventHypothesisReader);
