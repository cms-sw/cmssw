#include "L1Trigger/Phase2L1ParticleFlow/interface/L1NNTauProducer.hh"
#include <TLorentzVector.h>
#include <cmath>

L1NNTauProducer::L1NNTauProducer(const edm::ParameterSet& cfg)
    : fSeedPt_(cfg.getParameter<double>("seedpt")),
      fConeSize_(cfg.getParameter<double>("conesize")),
      fTauSize_(cfg.getParameter<double>("tausize")),
      fMaxTaus_(cfg.getParameter<int>("maxtaus")),
      fNParticles_(cfg.getParameter<int>("nparticles")),
      fL1PFToken_(consumes<vector<l1t::PFCandidate> >(cfg.getParameter<edm::InputTag>("L1PFObjects"))) {
  std::string lNNFile = cfg.getParameter<std::string>("NNFileName");  //,"L1Trigger/Phase2L1Taus/data/tau_3layer.pb");
  fTauNNId_ = std::make_unique<TauNNId>();
  if (lNNFile.find("v0") == std::string::npos)
    fTauNNId_->initialize("input_1:0", lNNFile, fNParticles_);
  else if (lNNFile.find("v0") != std::string::npos)
    fTauNNId_->initialize("dense_1_input:0", lNNFile, fNParticles_);
  produces<l1t::PFTauCollection>("L1PFTausNN");
}

void L1NNTauProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<l1t::PFCandidateCollection> l1PFCandidates;
  iEvent.getByToken(fL1PFToken_, l1PFCandidates);

  l1t::PFCandidateCollection pfChargedHadrons;
  l1t::PFCandidateCollection pfChargedHadrons_sort;
  l1t::PFCandidateCollection pfChargedHadrons_seeds;
  for (const auto& l1PFCand : *l1PFCandidates)
    if ((l1PFCand.id() == l1t::PFCandidate::ChargedHadron || l1PFCand.id() == l1t::PFCandidate::Electron) &&
        std::abs(l1PFCand.eta()) < 2.5)
      pfChargedHadrons_sort.push_back(l1PFCand);
  std::sort(pfChargedHadrons_sort.begin(), pfChargedHadrons_sort.end(), [](l1t::PFCandidate i, l1t::PFCandidate j) {
    return (i.pt() > j.pt());
  });
  auto lTaus = std::make_unique<l1t::PFTauCollection>();
  if (pfChargedHadrons_sort.empty()) {
    if (lTaus->empty()) {
      PFTau dummy;
      lTaus->push_back(dummy);
    }
    iEvent.put(std::move(lTaus), "L1PFTausNN");
    return;
  }
  pfChargedHadrons_seeds.push_back(pfChargedHadrons_sort[0]);
  for (unsigned int i0 = 1; i0 < pfChargedHadrons_sort.size(); i0++) {
    bool pMatch = false;
    for (unsigned int i1 = 0; i1 < pfChargedHadrons_seeds.size(); i1++) {
      if (deltaR(pfChargedHadrons_seeds[i1], pfChargedHadrons_sort[i0]) < fConeSize_)
        pMatch = true;
    }
    if (pMatch)
      continue;
    pfChargedHadrons_seeds.push_back(pfChargedHadrons_sort[i0]);
    if (int(pfChargedHadrons_seeds.size()) > fMaxTaus_ - 1)
      break;
  }
  for (unsigned int i0 = 0; i0 < pfChargedHadrons_seeds.size(); i0++) {
    addTau(pfChargedHadrons_seeds[i0], (*l1PFCandidates), lTaus);
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
  TLorentzVector lTot;
  lTot.SetPtEtaPhiM(0, 0, 0, 0);
  TLorentzVector lCand;
  lCand.SetPtEtaPhiM(0, 0, 0, 0);
  int lId = 0;
  for (auto l1PFCand : iParts) {
    if (deltaR(iCand, l1PFCand) > fConeSize_)
      continue;
    TLorentzVector pVec;
    pVec.SetPtEtaPhiM(l1PFCand.pt(), l1PFCand.eta(), l1PFCand.phi(), 0);
    lTot += pVec;
    if (deltaR(iCand, l1PFCand) < fTauSize_ &&
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
float L1NNTauProducer::deltaR(const l1t::PFCandidate& iPart1, const l1t::PFCandidate& iPart2) {
  float delta_r = 20;
  float pDPhi = fabs(iPart1.phi() - iPart2.phi());
  if (pDPhi > 2. * M_PI - pDPhi)
    pDPhi = 2. * M_PI - pDPhi;
  delta_r = sqrt((iPart1.eta() - iPart2.eta()) * (iPart1.eta() - iPart2.eta()) + pDPhi * pDPhi);
  return delta_r;
}
L1NNTauProducer::~L1NNTauProducer() {}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1NNTauProducer);
