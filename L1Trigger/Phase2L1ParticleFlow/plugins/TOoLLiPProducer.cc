#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/L1TParticleFlow/interface/PFJet.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/JetId.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/L1Trigger/interface/VertexWord.h"

#include <cmath>
#include <vector>

//HLS4ML compiled emulator modeling
#include <string>
#include "ap_fixed.h"
#include "hls4ml/emulator.h"

using namespace l1t;

class TOoLLiPProducer : public edm::stream::EDProducer<> {
public:
  explicit TOoLLiPProducer(const edm::ParameterSet&);
  ~TOoLLiPProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  std::unique_ptr<JetId> fJetId_;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<edm::View<l1t::PFJet>> const jets_;
  bool const fUseRawPt_;
  double const fMinPt_;
  double const fMaxEta_;
  unsigned int const fMaxJets_;
  int const fNParticles_;
  edm::EDGetTokenT<std::vector<l1t::VertexWord>> const fVtxEmu_;

  //HLS4ML emulator objects
  hls4mlEmulator::ModelLoader loader;
  std::shared_ptr<hls4mlEmulator::Model> model;
};

TOoLLiPProducer::TOoLLiPProducer(const edm::ParameterSet& cfg)
    : jets_(consumes<edm::View<l1t::PFJet>>(cfg.getParameter<edm::InputTag>("jets"))),
      fUseRawPt_(cfg.getParameter<bool>("useRawPt")),
      fMinPt_(cfg.getParameter<double>("minPt")),
      fMaxEta_(cfg.getParameter<double>("maxEta")),
      fMaxJets_(cfg.getParameter<int>("maxJets")),
      fNParticles_(cfg.getParameter<int>("nParticles")),
      fVtxEmu_(consumes<std::vector<l1t::VertexWord>>(cfg.getParameter<edm::InputTag>("vtx"))),
      loader(hls4mlEmulator::ModelLoader(cfg.getParameter<string>("TOoLLiPVersion"))) 
      {
  //load model and feed to JetID
  model = loader.load_model();
  fJetId_ = std::make_unique<JetId>(cfg.getParameter<std::string>("NNInput"), cfg.getParameter<std::string>("NNOutput"), model, fNParticles_);
  //produces<float>("L1LLPScores");
  produces<edm::ValueMap<float>>("L1PFLLPJets");
}

void TOoLLiPProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<edm::View<l1t::PFJet>> jets;
  iEvent.getByToken(jets_, jets);
  float vz = 0.;
  double ptsum = 0;
  edm::Handle<std::vector<l1t::VertexWord>> vtxEmuHandle;
  iEvent.getByToken(fVtxEmu_, vtxEmuHandle);
  for (const auto& vtx : *vtxEmuHandle) {
    if (ptsum == 0 || vtx.pt() > ptsum) {
      ptsum = vtx.pt();
      vz = vtx.z0();
    }
  }

 std::vector<float> LLPScores;
 for (const auto& srcjet : *jets) {
    if (((fUseRawPt_ ? srcjet.rawPt() : srcjet.pt()) < fMinPt_) || std::abs(srcjet.eta()) > fMaxEta_ ||
        LLPScores.size() >= fMaxJets_) {
      LLPScores.push_back(-1.);
      continue;
    }
    ap_fixed<16, 6> LLPScore = fJetId_->computeFixed(srcjet, vz, fUseRawPt_); 
    LLPScores.push_back(LLPScore);
  }

  auto outT = std::make_unique<edm::ValueMap<float>>();
  edm::ValueMap<float>::Filler fillerT(*outT);
  fillerT.insert(jets, LLPScores.begin(), LLPScores.end());
  fillerT.fill();

  iEvent.put(std::move(outT), "L1PFLLPJets");
}

void TOoLLiPProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jets", edm::InputTag("scPFL1Puppi"));
  desc.add<bool>("useRawPt", true);
  //change for LLP
  desc.add<std::string>("TOoLLiPVersion", std::string("/src/L1Trigger/Phase2L1ParticleFlow/test/TOoLLip_emulator_v1.so"));
  desc.add<std::string>("NNInput", "input:0");
  desc.add<std::string>("NNOutput", "sequential/dense_2/Sigmoid");
  desc.add<int>("maxJets", 10);
  desc.add<int>("nParticles", 10);
  desc.add<double>("minPt", 20);
  desc.add<double>("maxEta", 2.4);
  desc.add<edm::InputTag>("vtx", edm::InputTag("L1VertexFinderEmulator", "L1VerticesEmulation"));
  descriptions.add("TOoLLiPProducer", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TOoLLiPProducer);
