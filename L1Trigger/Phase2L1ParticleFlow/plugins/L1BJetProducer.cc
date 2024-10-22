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

using namespace l1t;

class L1BJetProducer : public edm::stream::EDProducer<edm::GlobalCache<BJetTFCache>> {
public:
  explicit L1BJetProducer(const edm::ParameterSet&, const BJetTFCache*);
  ~L1BJetProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  static std::unique_ptr<BJetTFCache> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(const BJetTFCache*);

private:
  std::unique_ptr<JetId> fBJetId_;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<edm::View<l1t::PFJet>> const jets_;
  bool const fUseRawPt_;
  double const fMinPt_;
  double const fMaxEta_;
  unsigned int const fMaxJets_;
  int const fNParticles_;
  edm::EDGetTokenT<std::vector<l1t::VertexWord>> const fVtxEmu_;
};

L1BJetProducer::L1BJetProducer(const edm::ParameterSet& cfg, const BJetTFCache* cache)
    : jets_(consumes<edm::View<l1t::PFJet>>(cfg.getParameter<edm::InputTag>("jets"))),
      fUseRawPt_(cfg.getParameter<bool>("useRawPt")),
      fMinPt_(cfg.getParameter<double>("minPt")),
      fMaxEta_(cfg.getParameter<double>("maxEta")),
      fMaxJets_(cfg.getParameter<int>("maxJets")),
      fNParticles_(cfg.getParameter<int>("nParticles")),
      fVtxEmu_(consumes<std::vector<l1t::VertexWord>>(cfg.getParameter<edm::InputTag>("vtx"))) {
  fBJetId_ = std::make_unique<JetId>(
      cfg.getParameter<std::string>("NNInput"), cfg.getParameter<std::string>("NNOutput"), cache, fNParticles_);
  produces<edm::ValueMap<float>>("L1PFBJets");
}
std::unique_ptr<BJetTFCache> L1BJetProducer::initializeGlobalCache(const edm::ParameterSet& cfg) {
  edm::FileInPath fp(cfg.getParameter<edm::FileInPath>("NNFileName"));
  auto cache = std::make_unique<BJetTFCache>(fp.fullPath());
  return cache;
}
void L1BJetProducer::globalEndJob(const BJetTFCache* cache) {}
void L1BJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
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

  std::vector<float> bScores;

  for (const auto& srcjet : *jets) {
    if (((fUseRawPt_ ? srcjet.rawPt() : srcjet.pt()) < fMinPt_) || std::abs(srcjet.eta()) > fMaxEta_ ||
        bScores.size() >= fMaxJets_) {
      bScores.push_back(-1.);
      continue;
    }
    float NN = fBJetId_->compute(srcjet, vz, fUseRawPt_);
    bScores.push_back(NN);
  }

  auto outT = std::make_unique<edm::ValueMap<float>>();
  edm::ValueMap<float>::Filler fillerT(*outT);
  fillerT.insert(jets, bScores.begin(), bScores.end());
  fillerT.fill();

  iEvent.put(std::move(outT), "L1PFBJets");
}

void L1BJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // L1BJetProducer
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jets", edm::InputTag("scPFL1Puppi"));
  desc.add<bool>("useRawPt", true);
  desc.add<edm::FileInPath>("NNFileName",
                            edm::FileInPath("L1Trigger/Phase2L1ParticleFlow/data/modelTT_PUP_Off_dXY_XYCut_Graph.pb"));
  desc.add<std::string>("NNInput", "input:0");
  desc.add<std::string>("NNOutput", "sequential/dense_2/Sigmoid");
  desc.add<int>("maxJets", 10);
  desc.add<int>("nParticles", 10);
  desc.add<double>("minPt", 20);
  desc.add<double>("maxEta", 2.4);
  desc.add<edm::InputTag>("vtx", edm::InputTag("L1VertexFinderEmulator", "L1VerticesEmulation"));
  descriptions.add("L1BJetProducer", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1BJetProducer);
