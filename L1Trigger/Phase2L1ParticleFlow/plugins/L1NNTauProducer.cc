#include "DataFormats/Math/interface/deltaR.h"
#include <cmath>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TParticleFlow/interface/PFTau.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/TauNNId.h"

using namespace l1t;

class L1NNTauProducer : public edm::stream::EDProducer<edm::GlobalCache<tensorflow::SessionCache>> {
public:
  explicit L1NNTauProducer(const edm::ParameterSet&, const tensorflow::SessionCache*);
  ~L1NNTauProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  static std::unique_ptr<tensorflow::SessionCache> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(const tensorflow::SessionCache*){};

private:
  std::unique_ptr<TauNNId> fTauNNId_;
  void addTau(const l1t::PFCandidate& iCand,
              const l1t::PFCandidateCollection& iParts,
              std::unique_ptr<PFTauCollection>& outputTaus);
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  double fSeedPt_;
  double fConeSize_;
  double fTauSize_;
  int fMaxTaus_;
  int fNParticles_;
  edm::EDGetTokenT<vector<l1t::PFCandidate>> fL1PFToken_;
};

static constexpr float track_trigger_eta_max = 2.5;

L1NNTauProducer::L1NNTauProducer(const edm::ParameterSet& cfg, const tensorflow::SessionCache* cache)
    : fSeedPt_(cfg.getParameter<double>("seedpt")),
      fConeSize_(cfg.getParameter<double>("conesize")),
      fTauSize_(cfg.getParameter<double>("tausize")),
      fMaxTaus_(cfg.getParameter<int>("maxtaus")),
      fNParticles_(cfg.getParameter<int>("nparticles")),
      fL1PFToken_(consumes<vector<l1t::PFCandidate>>(cfg.getParameter<edm::InputTag>("L1PFObjects"))) {
  std::string lNNFile = cfg.getParameter<std::string>("NNFileName");  //,"L1Trigger/Phase2L1Taus/data/tau_3layer.pb");
  fTauNNId_ = std::make_unique<TauNNId>(lNNFile.find("v0") == std::string::npos ? "input_1:0" : "dense_1_input:0",
                                        cache->getSession(),
                                        lNNFile,
                                        fNParticles_);
  produces<l1t::PFTauCollection>("L1PFTausNN");
}

std::unique_ptr<tensorflow::SessionCache> L1NNTauProducer::initializeGlobalCache(const edm::ParameterSet& cfg) {
  tensorflow::setLogging("3");
  std::string graphPath = edm::FileInPath(cfg.getParameter<std::string>("NNFileName")).fullPath();
  return std::make_unique<tensorflow::SessionCache>(graphPath);
}

void L1NNTauProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<l1t::PFCandidateCollection> l1PFCandidates;
  iEvent.getByToken(fL1PFToken_, l1PFCandidates);

  std::vector<unique_ptr<l1t::PFCandidate>> pfChargedHadrons_sort_v;
  std::vector<unique_ptr<l1t::PFCandidate>> pfChargedHadrons_seeds_v;
  for (const auto& l1PFCand : *l1PFCandidates)
    if ((l1PFCand.id() == l1t::PFCandidate::ChargedHadron || l1PFCand.id() == l1t::PFCandidate::Electron) &&
        std::abs(l1PFCand.eta()) < track_trigger_eta_max)
      pfChargedHadrons_sort_v.push_back(std::make_unique<l1t::PFCandidate>(l1PFCand));

  std::sort(
      pfChargedHadrons_sort_v.begin(),
      pfChargedHadrons_sort_v.end(),
      [](std::unique_ptr<l1t::PFCandidate>& i, std::unique_ptr<l1t::PFCandidate>& j) { return (i->pt() > j->pt()); });

  auto lTaus = std::make_unique<l1t::PFTauCollection>();
  if (pfChargedHadrons_sort_v.empty()) {
    if (lTaus->empty()) {
      PFTau dummy;
      lTaus->push_back(dummy);
    }
    iEvent.put(std::move(lTaus), "L1PFTausNN");
    return;
  }
  pfChargedHadrons_seeds_v.push_back(std::move(pfChargedHadrons_sort_v[0]));
  for (unsigned int i0 = 1; i0 < pfChargedHadrons_sort_v.size(); i0++) {
    bool pMatch = false;
    for (unsigned int i1 = 0; i1 < pfChargedHadrons_seeds_v.size(); i1++) {
      if (reco::deltaR2(*(pfChargedHadrons_seeds_v[i1]), *(pfChargedHadrons_sort_v[i0])) < fConeSize_ * fConeSize_)
        pMatch = true;
    }
    if (pMatch)
      continue;
    pfChargedHadrons_seeds_v.push_back(std::move(pfChargedHadrons_sort_v[i0]));
    if (int(pfChargedHadrons_seeds_v.size()) > fMaxTaus_ - 1)
      break;
  }
  for (unsigned int i0 = 0; i0 < pfChargedHadrons_seeds_v.size(); i0++) {
    addTau(*(pfChargedHadrons_seeds_v[i0]), (*l1PFCandidates), lTaus);
  }
  if (lTaus->empty()) {
    PFTau dummy;
    lTaus->push_back(dummy);
  }
  std::sort(lTaus->begin(), lTaus->end(), [](l1t::PFTau i, l1t::PFTau j) { return (i.pt() > j.pt()); });
  iEvent.put(std::move(lTaus), "L1PFTausNN");
}

// create taus based on grid structure
void L1NNTauProducer::addTau(const l1t::PFCandidate& iCand,
                             const l1t::PFCandidateCollection& iParts,
                             std::unique_ptr<l1t::PFTauCollection>& outputTaus) {
  l1t::PFCandidateCollection pfTauCands;
  math::PtEtaPhiMLorentzVector lTot(0, 0, 0, 0);
  math::PtEtaPhiMLorentzVector lCand(0, 0, 0, 0);
  int lId = 0;
  for (const auto& l1PFCand : iParts) {
    if (reco::deltaR2(iCand, l1PFCand) > fConeSize_ * fConeSize_)
      continue;
    math::PtEtaPhiMLorentzVector pVec(l1PFCand.pt(), l1PFCand.eta(), l1PFCand.phi(), 0);
    lTot += pVec;
    if (reco::deltaR2(iCand, l1PFCand) < fTauSize_ * fTauSize_ &&
        (l1PFCand.id() == l1t::PFCandidate::Electron || l1PFCand.id() == l1t::PFCandidate::ChargedHadron ||
         l1PFCand.id() == l1t::PFCandidate::Photon)) {
      lId++;
      lCand += pVec;
    }
    pfTauCands.push_back(l1PFCand);
  }
  if (lTot.Pt() < fSeedPt_)
    return;
  std::sort(
      pfTauCands.begin(), pfTauCands.end(), [](l1t::PFCandidate i, l1t::PFCandidate j) { return (i.pt() > j.pt()); });
  float NN = fTauNNId_->compute(iCand, pfTauCands);
  math::PtEtaPhiMLorentzVector tempP4(lCand.Pt(), lCand.Eta(), lCand.Phi(), lCand.M());
  l1t::PFTau l1PFTau(tempP4, NN, 0, lId);
  outputTaus->push_back(l1PFTau);
}
void L1NNTauProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // L1NNTauProducer
  edm::ParameterSetDescription desc;
  desc.add<std::string>("NNFileName", "L1Trigger/Phase2L1ParticleFlow/data/tau_3layer.pb");
  desc.add<double>("tausize", 0.1);
  desc.add<int>("maxtaus", 5);
  desc.add<int>("nparticles", 10);
  desc.add<double>("conesize", 0.4);
  desc.add<double>("seedpt", 20);
  desc.add<edm::InputTag>("L1PFObjects", edm::InputTag("L1PFProducer", "l1pfCandidates"));
  descriptions.add("L1NNTauProducer", desc);
}
L1NNTauProducer::~L1NNTauProducer() {}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1NNTauProducer);
