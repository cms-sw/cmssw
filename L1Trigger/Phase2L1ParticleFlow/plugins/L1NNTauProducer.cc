#include "DataFormats/Math/interface/deltaR.h"
#include <TLorentzVector.h>
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
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/TauNNIdHW.h"

#include "ap_int.h"
#include "ap_fixed.h"

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
  std::unique_ptr<TauNNIdHW> fTauNNIdHW_;

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void process_SW(const l1t::PFCandidateCollection& parts, std::unique_ptr<l1t::PFTauCollection>& iTaus);
  void process_HW(const l1t::PFCandidateCollection& parts, std::unique_ptr<l1t::PFTauCollection>& iTaus);
  void makeTau_HW(const l1t::PFCandidate& seed,
                  l1t::PFCandidateCollection& parts,
                  std::unique_ptr<l1t::PFTauCollection>& iTaus);

  void addTau(const l1t::PFCandidate& iCand,
              const l1t::PFCandidateCollection& iParts,
              std::unique_ptr<PFTauCollection>& outputTaus);

  double fSeedPt_;
  double fConeSize_;
  double fTauSize_;
  int fMaxTaus_;
  int fNParticles_;
  bool fHW;
  bool fEMSeed;
  bool fDebug;
  edm::EDGetTokenT<vector<l1t::PFCandidate>> fL1PFToken_;
};

static constexpr float track_trigger_eta_max = 2.5;

L1NNTauProducer::L1NNTauProducer(const edm::ParameterSet& cfg, const tensorflow::SessionCache* cache)
    : fSeedPt_(cfg.getParameter<double>("seedpt")),
      fConeSize_(cfg.getParameter<double>("conesize")),
      fTauSize_(cfg.getParameter<double>("tausize")),
      fMaxTaus_(cfg.getParameter<int>("maxtaus")),
      fNParticles_(cfg.getParameter<int>("nparticles")),
      fHW(cfg.getParameter<bool>("HW")),
      fEMSeed(cfg.getParameter<bool>("emseed")),
      fDebug(cfg.getParameter<bool>("debug")),
      fL1PFToken_(consumes<vector<l1t::PFCandidate>>(cfg.getParameter<edm::InputTag>("L1PFObjects"))) {
  std::string lNNFile = cfg.getParameter<std::string>("NNFileName");  //,"L1Trigger/Phase2L1Taus/data/tau_3layer.pb");
  if (fHW) {
    fTauNNIdHW_ = std::make_unique<TauNNIdHW>();
    fTauNNIdHW_->initialize("input_1:0", fNParticles_);
  } else {
    fTauNNId_ = std::make_unique<TauNNId>(lNNFile.find("v0") == std::string::npos ? "input_1:0" : "dense_1_input:0",
                                          cache->getSession(),
                                          lNNFile,
                                          fNParticles_);
  }
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
  auto lTaus = std::make_unique<l1t::PFTauCollection>();

  if (fHW) {
    process_HW(*l1PFCandidates, lTaus);
  } else {
    process_SW(*l1PFCandidates, lTaus);
  }

  if (lTaus->empty()) {
    PFTau dummy;
    lTaus->push_back(dummy);
  }
  std::sort(lTaus->begin(), lTaus->end(), [](l1t::PFTau i, l1t::PFTau j) { return (i.pt() > j.pt()); });
  iEvent.put(std::move(lTaus), "L1PFTausNN");
}
void L1NNTauProducer::process_SW(const l1t::PFCandidateCollection& parts,
                                 std::unique_ptr<l1t::PFTauCollection>& iTaus) {
  std::vector<unique_ptr<l1t::PFCandidate>> pfChargedHadrons;
  std::vector<unique_ptr<l1t::PFCandidate>> pfChargedHadrons_sort_v;
  std::vector<unique_ptr<l1t::PFCandidate>> pfChargedHadrons_seeds_v;
  for (const auto& l1PFCand : parts)
    if ((l1PFCand.id() == l1t::PFCandidate::ChargedHadron || l1PFCand.id() == l1t::PFCandidate::Electron) &&
        std::abs(l1PFCand.eta()) < track_trigger_eta_max)
      pfChargedHadrons_sort_v.push_back(std::make_unique<l1t::PFCandidate>(l1PFCand));

  if (pfChargedHadrons_sort_v.empty())
    return;
  std::sort(
      pfChargedHadrons_sort_v.begin(),
      pfChargedHadrons_sort_v.end(),
      [](std::unique_ptr<l1t::PFCandidate>& i, std::unique_ptr<l1t::PFCandidate>& j) { return (i->pt() > j->pt()); });

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
    addTau(*(pfChargedHadrons_seeds_v[i0]), parts, iTaus);
  }
}

// create taus based on grid structure
void L1NNTauProducer::addTau(const l1t::PFCandidate& iCand,
                             const l1t::PFCandidateCollection& iParts,
                             std::unique_ptr<l1t::PFTauCollection>& outputTaus) {
  l1t::PFCandidateCollection pfTauCands;
  math::PtEtaPhiMLorentzVector lTot(0, 0, 0, 0);
  math::PtEtaPhiMLorentzVector lCand(0, 0, 0, 0);
  int lId = 0;
  float z0 = 0;
  float dxy = 0;
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
      if (z0 == 0 && l1PFCand.id() == l1t::PFCandidate::ChargedHadron) {
        z0 = l1PFCand.z0();
        dxy = l1PFCand.dxy();
      }
    }
    pfTauCands.push_back(l1PFCand);
  }
  if (lTot.Pt() < fSeedPt_)
    return;
  std::sort(
      pfTauCands.begin(), pfTauCands.end(), [](l1t::PFCandidate i, l1t::PFCandidate j) { return (i.pt() > j.pt()); });
  float NN = fTauNNId_->compute(iCand, pfTauCands);
  float* lNNVector = fTauNNId_->NNVectorVar();
  math::PtEtaPhiMLorentzVector tempP4(lCand.Pt(), lCand.Eta(), lCand.Phi(), lCand.M() * lCand.M());
  l1t::PFTau l1PFTau(tempP4, lNNVector, NN, 0, lId);
  l1PFTau.setZ0(z0);
  l1PFTau.setDxy(dxy);
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
  desc.add<bool>("HW", true);
  desc.add<bool>("emseed", true);
  desc.add<bool>("debug", false);
  desc.add<edm::InputTag>("L1PFObjects", edm::InputTag("L1PFProducer", "l1pfCandidates"));
  descriptions.add("L1NNTauProducer", desc);
}

void L1NNTauProducer::makeTau_HW(const l1t::PFCandidate& seed,
                                 l1t::PFCandidateCollection& parts,
                                 std::unique_ptr<l1t::PFTauCollection>& iTaus) {
  // Seed Cone Jet algorithm with ap_fixed types and hardware emulation
  L1TauEmu::detaphi_t rCone2 =
      L1TauEmu::detaphi_t(fTauSize_ * fTauSize_ * L1TauEmu::etaphi_base * L1TauEmu::etaphi_base);
  unsigned lId = 0;
  L1TauEmu::pt_t pt_tot = 0;
  input2_t p1_tot = 0;
  input2_t p1x_tot = 0;
  input2_t p1y_tot = 0;
  input2_t p1z_tot = 0;

  float p_tot = 0;
  float px_tot = 0;
  float py_tot = 0;
  float pz_tot = 0;

  float eta_1 = seed.eta();
  float phi_1 = seed.phi();
  input_t e1ta_1 = seed.eta();
  input_t p1hi_1 = seed.phi();
  L1TauEmu::pt_t pt = 0;
  L1TauEmu::z0_t z0 = 0;
  L1TauEmu::dxy_t dxy = 0;
  for (unsigned i0 = 0; i0 < parts.size(); i0++) {
    pt_tot = pt_tot + L1TauEmu::pt_t(parts[i0].pt());
    if (L1TauEmu::inCone(seed, (parts[i0]), rCone2)) {
      if (parts[i0].id() == l1t::PFCandidate::Electron || parts[i0].id() == l1t::PFCandidate::ChargedHadron ||
          parts[i0].id() == l1t::PFCandidate::Photon) {
        lId++;
        pt = pt + L1TauEmu::pt_t(parts[i0].pt());
        float deta = parts[i0].eta() - eta_1;
        float dphi = parts[i0].phi() - phi_1;
        float dr2 = deta * deta + dphi * dphi;
        pz_tot = pz_tot + (parts[i0].pt()) * (1 - dr2 * 0.5);
        py_tot = py_tot + (parts[i0].pt()) * dphi;  //sin(dphi ));
        px_tot = px_tot + (parts[i0].pt()) * deta;  //sin(deta ));

        input2_t d1eta = input_t(parts[i0].eta()) - e1ta_1;
        input2_t d1phi = input_t(parts[i0].phi()) - p1hi_1;
        input2_t d1r2 = d1eta * d1eta + d1phi * d1phi;
        input2_t tmppt = input_t(parts[i0].pt());
        input2_t half = 0.5;
        p1z_tot = p1z_tot + tmppt * (1 - d1r2 * half);
        p1y_tot = p1y_tot + tmppt * d1phi;
        p1x_tot = p1x_tot + tmppt * d1eta;
        p_tot = p_tot + (parts[i0].pt());
        p1_tot = p1_tot + tmppt;
        if (z0 == 0 && parts[i0].id() == l1t::PFCandidate::ChargedHadron) {
          z0 = parts[i0].hwZ0();
          dxy = parts[i0].hwDxy();
        }
      }
    }
  }
  input2_t tmpmass1 = (p1_tot * p1_tot - p1x_tot * p1x_tot - p1y_tot * p1y_tot - p1z_tot * p1z_tot);
  if (tmpmass1 < 0)
    tmpmass1 = 0;
  L1TauEmu::pt_t mass = l1ct::pt_t(tmpmass1);
  if (pt < fSeedPt_)
    return;
  result_t NN = fTauNNIdHW_->compute(seed, parts);
  input_t* lNNVector = fTauNNIdHW_->NNVectorVar();
  float pNNVec[80];
  for (unsigned i0 = 0; i0 < 80; i0++)
    pNNVec[i0] = float(lNNVector[i0]);
  L1TauEmu::etaphi_t eta = etaphi_t(seed.eta() * L1TauEmu::etaphi_base);
  L1TauEmu::etaphi_t phi = etaphi_t(seed.phi() * L1TauEmu::etaphi_base);

  //Firmware Tau
  l1ct::Tau l1ctTau;
  l1ctTau.hwPt = l1ct::pt_t(pt);  //l1gt is <16,11> and currently <16,14>
  l1ctTau.hwEta = l1ct::Scales::makeGlbEta(float(eta));
  l1ctTau.hwPhi = l1ct::Scales::makeGlbPhi(float(phi));

  l1ctTau.hwSeedPt = seed.pt();
  l1ctTau.hwSeedZ0 = seed.hwZ0();
  l1ctTau.hwCharge = seed.charge();

  l1ctTau.hwType = l1ct::Tau::type_t(lId);
  l1ctTau.hwRawId = ap_uint<10>(NN * 1024);  //NN Output is ap_fixed<16, 8> so need to cast.

  //Convert to GT format and pack to encodedTau of PFTau
  l1gt::Tau l1gtTau = l1ctTau.toGT();
  l1gt::PackedTau packed_Tau = l1gtTau.pack();

  //Make PFTau
  //Save pt, eta and phi in gt scales
  math::PtEtaPhiMLorentzVector tempP4(l1gt::Scales::floatPt(l1gtTau.v3.pt),
                                      l1gt::Scales::floatEta(l1gtTau.v3.eta),
                                      l1gt::Scales::floatPhi(l1gtTau.v3.phi),
                                      float(mass));

  l1t::PFTau l1PFTau(tempP4, pNNVec, NN, 0, lId);
  l1PFTau.setZ0(float(z0) * 0.05);    //L1TauEmu::z0_base);
  l1PFTau.setDxy(float(dxy) * 0.05);  //L1TauEmu::dxy_base);

  l1PFTau.set_encodedTau(packed_Tau);

  iTaus->push_back(l1PFTau);
}

void L1NNTauProducer::process_HW(const l1t::PFCandidateCollection& parts,
                                 std::unique_ptr<l1t::PFTauCollection>& iTaus) {
  // The fixed point algorithm emulation
  using namespace L1TauEmu;
  std::vector<l1t::PFCandidate> work;
  work.resize(parts.size());
  std::transform(parts.begin(), parts.end(), work.begin(), [](const l1t::PFCandidate& part) { return part; });
  std::sort(work.begin(), work.end(), [](l1t::PFCandidate i, l1t::PFCandidate j) {
    return (l1ct::pt_t(i.pt()) > l1ct::pt_t(j.pt()));
  });

  std::vector<l1t::PFCandidate> seeds;
  uint lSeed = l1t::PFCandidate::ChargedHadron;
  if (fEMSeed)
    lSeed = l1t::PFCandidate::Photon;
  std::copy_if(work.begin(), work.end(), std::back_inserter(seeds), [&](const l1t::PFCandidate& part) {
    return ((part.id() == l1t::PFCandidate::Electron || part.id() == l1t::PFCandidate::ChargedHadron ||
             part.id() == lSeed) &&
            std::abs(part.eta()) < track_trigger_eta_max);
  });
  // It would be nice to transform the inputs to the etaphi_base of the FW here, as in the line below
  // However the phi may wrap around if the etaphi_base > 1, so don't do it...
  //std::for_each(work.begin(), work.end(), [](l1t::PFCandidate& x){x.setP4(math::PtEtaPhiMLorentzVector(pt_t(x.pt()), etaphi_t(x.eta()*etaphi_base), etaphi_t(x.phi()*etaphi_base), x.mass()));});
  detaphi_t rCone2 = detaphi_t(fConeSize_ * fConeSize_ * etaphi_base * etaphi_base);

  iTaus->reserve(fMaxTaus_);
  while (!seeds.empty() && iTaus->size() < unsigned(fMaxTaus_)) {
    // Take the first (highest pt) candidate as a seed
    l1t::PFCandidate seed = seeds.at(0);
    // Get the particles within a _coneSize of the seed
    std::vector<l1t::PFCandidate> particlesInCone;
    std::copy_if(work.begin(), work.end(), std::back_inserter(particlesInCone), [&](l1t::PFCandidate& part) {
      return inCone(seed, part, rCone2);
    });
    makeTau_HW(seed, particlesInCone, iTaus);
    // remove the clustered particles
    work.erase(std::remove_if(
                   work.begin(), work.end(), [&](const l1t::PFCandidate& part) { return inCone(seed, part, rCone2); }),
               work.end());

    seeds.erase(
        std::remove_if(
            seeds.begin(), seeds.end(), [&](const l1t::PFCandidate& part) { return inCone(seed, part, rCone2); }),
        seeds.end());
  }
}

L1NNTauProducer::~L1NNTauProducer() {}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1NNTauProducer);
