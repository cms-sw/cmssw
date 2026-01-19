/** \class HLTADFilter
 * HLT filter using PyTorch anomaly detection model.
 * Author: Maciej Glowacki
 */

#include <algorithm>
#include <map>
#include <torch/torch.h>
#include <vector>
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/Math/interface/libminifloat.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/PyTorch/interface/Model.h"

namespace {
  constexpr size_t kMaxObjects = 500;
  constexpr size_t kNFeatures = 6;

  struct UnifiedParticle {
    float pt, eta, phi, dxy, dxysig, pid;
    UnifiedParticle(float p, float e, float ph, float dx, float ds, float id)
        : pt(p), eta(e), phi(ph), dxy(dx), dxysig(ds), pid(id) {}
  };

  void sortParticlesByPt(std::vector<UnifiedParticle>& particles) {
    std::partial_sort(particles.begin(),
                      particles.begin() + std::min(particles.size(), kMaxObjects),
                      particles.end(),
                      [](const UnifiedParticle& a, const UnifiedParticle& b) { return a.pt > b.pt; });
  }
}  // namespace

class HLTADFilter : public edm::stream::EDFilter<> {
public:
  explicit HLTADFilter(const edm::ParameterSet& cfg)
      : pfToken_(consumes<std::vector<reco::PFCandidate>>(cfg.getParameter<edm::InputTag>("pfCandidates"))),
        muonToken_(consumes<reco::MuonCollection>(cfg.getParameter<edm::InputTag>("muons"))),
        egammaCandToken_(consumes<reco::RecoEcalCandidateCollection>(cfg.getParameter<edm::InputTag>("egammaCands"))),
        gsfTrackToken_(consumes<reco::GsfTrackCollection>(cfg.getParameter<edm::InputTag>("gsfTracks"))),
        vertexToken_(consumes<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("vertices"))),
        tensorBuffer_(kMaxObjects * kNFeatures, 0.0f),
        threshold_(cfg.getParameter<double>("threshold")) {
    edm::FileInPath modelFilePath = cfg.getParameter<edm::FileInPath>("modelPath");
    model_ = std::make_unique<cms::torch::Model>(modelFilePath.fullPath(), torch::Device(torch::kCPU));
    InputTensor_ = torch::from_blob(
        tensorBuffer_.data(), {1, static_cast<long>(kMaxObjects), static_cast<long>(kNFeatures)}, torch::kFloat32);
  }

  bool filter(edm::Event& event, const edm::EventSetup&) override {
    std::vector<UnifiedParticle> allParticles;
    allParticles.reserve(kMaxObjects);

    const auto& vertexCollection = event.getHandle(vertexToken_);
    const reco::Vertex* pv =
        (vertexCollection.isValid() && !vertexCollection->empty()) ? &((*vertexCollection)[0]) : nullptr;

    // --- 1. Collect PF Candidates ---
    edm::Handle<std::vector<reco::PFCandidate>> pfH;
    if (event.getByToken(pfToken_, pfH)) {
      for (const auto& cand : *pfH) {
        if (cand.pdgId() == reco::PFCandidate::h || cand.pdgId() == reco::PFCandidate::e)
          continue;

        float dxy = 0, dxysig = 0;
        if (const auto* trk = cand.bestTrack()) {
          if (pv) {
            dxy = trk->dxy(pv->position());
            if (trk->dxyError() > 0)
              dxysig = dxy / trk->dxyError();
            dxysig = MiniFloatConverter::reduceMantissaToNbitsRounding(dxysig, 10);
          }
        }
        allParticles.emplace_back(cand.pt(), cand.eta(), cand.phi(), dxy, dxysig, static_cast<float>(cand.pdgId()));
      }
    }

    // --- 2. Collect Muons ---
    edm::Handle<std::vector<reco::Muon>> muH;
    if (event.getByToken(muonToken_, muH)) {
      for (const auto& m : *muH) {
        float dxy = 0, dxysig = 0;
        auto trk = m.innerTrack();
        if (trk.isNonnull() && pv) {
          dxy = trk->dxy(pv->position());
          if (trk->dxyError() > 0)
            dxysig = dxy / trk->dxyError();
          dxysig = MiniFloatConverter::reduceMantissaToNbitsRounding(dxysig, 10);
        }
        allParticles.emplace_back(m.pt(), m.eta(), m.phi(), dxy, dxysig, 13.0f);
      }
    }

    // --- 3. Collect EGamma ---
    edm::Handle<reco::GsfTrackCollection> gsfTrkH;
    event.getByToken(gsfTrackToken_, gsfTrkH);

    edm::Handle<reco::RecoEcalCandidateCollection> egammaH;
    event.getByToken(egammaCandToken_, egammaH);

    // Build map from SuperCluster to GsfTrack
    std::map<reco::SuperClusterRef, const reco::GsfTrack*> scToTrack;
    for (const auto& trk : *gsfTrkH) {
      if (trk.extra().isNonnull() && trk.extra()->seedRef().isNonnull()) {
        auto elseed = trk.extra()->seedRef().castTo<reco::ElectronSeedRef>();
        auto sc = elseed->caloCluster().castTo<reco::SuperClusterRef>();
        scToTrack[sc] = &trk;
      }
    }

    // Loop over EGamma candidates
    for (const auto& cand : *egammaH) {
      float dxy = 0, dxysig = 0, pid = 22.0f;
      reco::SuperClusterRef candSC = cand.superCluster();

      auto it = scToTrack.find(candSC);
      if (it != scToTrack.end() && pv) {
        const reco::GsfTrack* trk = it->second;
        dxy = trk->dxy(pv->position());
        if (trk->dxyError() > 0)
          dxysig = dxy / trk->dxyError();
        dxysig = MiniFloatConverter::reduceMantissaToNbitsRounding(dxysig, 10);

        pid = (trk->charge() < 0) ? 11.0f : -11.0f;
      }

      allParticles.emplace_back(cand.pt(), cand.eta(), cand.phi(), dxy, dxysig, pid);
    }

    sortParticlesByPt(allParticles);
    std::fill(tensorBuffer_.begin(), tensorBuffer_.end(), 0.0f);

    size_t nToFill = std::min(allParticles.size(), kMaxObjects);
    for (size_t i = 0; i < nToFill; ++i) {
      const auto& p = allParticles[i];
      tensorBuffer_[i * kNFeatures + 0] = p.pt;
      tensorBuffer_[i * kNFeatures + 1] = p.eta;
      tensorBuffer_[i * kNFeatures + 2] = p.phi;
      tensorBuffer_[i * kNFeatures + 3] = p.dxy;
      tensorBuffer_[i * kNFeatures + 4] = p.dxysig;
      tensorBuffer_[i * kNFeatures + 5] = p.pid;
    }

    torch::NoGradGuard noGrad;
    std::vector<torch::IValue> inputs;
    inputs.push_back(InputTensor_);

    at::Tensor output = model_->forward(inputs).toTensor();
    bool accept = output.item<float>() > threshold_;

    return accept;
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("pfCandidates", edm::InputTag("hltParticleFlow"));
    desc.add<edm::InputTag>("muons", edm::InputTag("hltIterL3MuonsNoVtx"));
    desc.add<edm::InputTag>("egammaCands", edm::InputTag("hltEgammaCandidates"));
    desc.add<edm::InputTag>("gsfTracks", edm::InputTag("hltEgammaGsfTracks"));
    desc.add<edm::InputTag>("vertices", edm::InputTag("hltPixelVertices"));
    desc.add<edm::FileInPath>("modelPath", edm::FileInPath("HLTrigger/HLTfilters/data/hlt_ad_model.pt"));
    desc.add<double>("threshold", 21.499275);
    descriptions.add("hltADFilter", desc);
  }

private:
  const edm::EDGetTokenT<std::vector<reco::PFCandidate>> pfToken_;
  const edm::EDGetTokenT<reco::MuonCollection> muonToken_;
  const edm::EDGetTokenT<reco::RecoEcalCandidateCollection> egammaCandToken_;
  const edm::EDGetTokenT<reco::GsfTrackCollection> gsfTrackToken_;
  const edm::EDGetTokenT<reco::VertexCollection> vertexToken_;

  std::unique_ptr<cms::torch::Model> model_;
  std::vector<float> tensorBuffer_;
  torch::Tensor InputTensor_;
  const double threshold_;
};

DEFINE_FWK_MODULE(HLTADFilter);
