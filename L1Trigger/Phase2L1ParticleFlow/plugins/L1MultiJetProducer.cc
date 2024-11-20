#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/L1TParticleFlow/interface/PFJet.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/MultiJetId.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/L1Trigger/interface/VertexWord.h"

#include <cmath>
#include <vector>

#include <string>
#include "ap_fixed.h"
#include "hls4ml/emulator.h"

using namespace l1t;

class L1MultiJetProducer : public edm::stream::EDProducer<> {
public:
  explicit L1MultiJetProducer(const edm::ParameterSet&);
  ~L1MultiJetProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  std::unique_ptr<MultiJetId> fJetId_;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<edm::View<l1t::PFJet>> const jets_;
  bool const fUseRawPt_;
  double const fMinPt_;
  double const fMaxEta_;
  unsigned int const fMaxJets_;
  int const fNParticles_;

  hls4mlEmulator::ModelLoader loader;
  std::shared_ptr<hls4mlEmulator::Model> model;
};

L1MultiJetProducer::L1MultiJetProducer(const edm::ParameterSet& cfg)
    : jets_(consumes<edm::View<l1t::PFJet>>(cfg.getParameter<edm::InputTag>("jets"))),
      fUseRawPt_(cfg.getParameter<bool>("useRawPt")),
      fMinPt_(cfg.getParameter<double>("minPt")),
      fMaxEta_(cfg.getParameter<double>("maxEta")),
      fMaxJets_(cfg.getParameter<int>("maxJets")),
      fNParticles_(cfg.getParameter<int>("nParticles")),
      loader(hls4mlEmulator::ModelLoader(cfg.getParameter<string>("MultiJetPath"))) {
  model = loader.load_model();
  fJetId_ = std::make_unique<MultiJetId>(model, fNParticles_);
  produces<edm::ValueMap<std::vector<float>>>("L1PFMultiJets");
}

void L1MultiJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<edm::View<l1t::PFJet>> jets;
  iEvent.getByToken(jets_, jets);

  std::vector<std::vector<float>> jetScores;

  for (const auto& srcjet : *jets) {
    if (((fUseRawPt_ ? srcjet.rawPt() : srcjet.pt()) < fMinPt_) || std::abs(srcjet.eta()) > fMaxEta_ ||
        jetScores.size() >= fMaxJets_) {
      jetScores.push_back({-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.});
      continue;
    }
    std::vector<float> JetScore_float = fJetId_->computeFixed(srcjet, fUseRawPt_);
    jetScores.push_back(JetScore_float);
  }

  auto outT = std::make_unique<edm::ValueMap<std::vector<float>>>();
  edm::ValueMap<std::vector<float>>::Filler fillerT(*outT);
  fillerT.insert(jets, jetScores.begin(), jetScores.end());
  fillerT.fill();
  iEvent.put(std::move(outT), "L1PFMultiJets");
}


void L1MultiJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jets", edm::InputTag("scPFL1Puppi"));
  desc.add<bool>("useRawPt", true);
  desc.add<std::string>("MultiJetPath", std::string("L1Trigger/Phase2L1ParticleFlow/data/MultiJetBaseline"));
  desc.add<int>("maxJets", 16);
  desc.add<int>("nParticles", 16);
  desc.add<double>("minPt", 20);
  desc.add<double>("maxEta", 2.4);
  desc.add<edm::InputTag>("vtx", edm::InputTag("L1VertexFinderEmulator", "L1VerticesEmulation"));
  descriptions.add("L1MultiJetProducer", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1MultiJetProducer);
