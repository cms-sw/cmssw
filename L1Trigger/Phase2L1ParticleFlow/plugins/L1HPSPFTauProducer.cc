#include <vector>
#include <numeric>

////////////////////
// FRAMEWORK HEADERS
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "DataFormats/L1TParticleFlow/interface/PFTau.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
// bitwise emulation headers
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/L1HPSPFTauEmulator.h"

class L1HPSPFTauProducer : public edm::global::EDProducer<> {
public:
  explicit L1HPSPFTauProducer(const edm::ParameterSet&);
  ~L1HPSPFTauProducer() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  //various needed vars
  int nTaus_;
  bool HW_;
  bool fUseJets_;
  bool debug_;

  edm::EDGetTokenT<l1t::PFCandidateCollection> tokenL1PFCands_;
  //jets
  edm::EDGetTokenT<std::vector<reco::CaloJet>> tokenL1PFJets_;
  //functions
  std::vector<l1t::PFTau> processEvent_HW(std::vector<edm::Ptr<l1t::PFCandidate>>& parts,
                                          std::vector<edm::Ptr<reco::CaloJet>>& jets) const;

  static std::vector<l1HPSPFTauEmu::Particle> convertJetsToHW(std::vector<edm::Ptr<reco::CaloJet>>& edmJets);
  static std::vector<l1HPSPFTauEmu::Particle> convertEDMToHW(std::vector<edm::Ptr<l1t::PFCandidate>>& edmParticles);
  static std::vector<l1t::PFTau> convertHWToEDM(std::vector<l1HPSPFTauEmu::Tau> hwTaus);
  //ADD
  edm::EDPutTokenT<l1t::PFTauCollection> tauToken_;
};

L1HPSPFTauProducer::L1HPSPFTauProducer(const edm::ParameterSet& cfg)
    : nTaus_(cfg.getParameter<int>("nTaus")),
      HW_(cfg.getParameter<bool>("HW")),
      fUseJets_(cfg.getParameter<bool>("useJets")),
      debug_(cfg.getParameter<bool>("debug")),
      tokenL1PFCands_(consumes(cfg.getParameter<edm::InputTag>("srcL1PFCands"))),
      tokenL1PFJets_(consumes(cfg.getParameter<edm::InputTag>("srcL1PFJets"))),
      tauToken_{produces()} {}

void L1HPSPFTauProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("srcL1PFCands", edm::InputTag("l1tLayer1", "Puppi"));
  desc.add<int>("nTaus", 16);
  desc.add<bool>("HW", true);
  desc.add<bool>("useJets", false);
  desc.add<bool>("debug", false);
  desc.add<edm::InputTag>("srcL1PFJets",
                          edm::InputTag("l1tPhase1JetCalibrator9x9trimmed", "Phase1L1TJetFromPfCandidates"));
  descriptions.add("l1tHPSPFTauProducer", desc);
}

void L1HPSPFTauProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto l1PFCandidates = iEvent.getHandle(tokenL1PFCands_);
  //add jets even if not used, for simplicity
  auto l1PFJets = iEvent.getHandle(tokenL1PFJets_);
  //

  //adding collection
  std::vector<edm::Ptr<l1t::PFCandidate>> particles;
  particles.reserve((*l1PFCandidates).size());
  for (unsigned i = 0; i < (*l1PFCandidates).size(); i++) {
    particles.push_back(edm::Ptr<l1t::PFCandidate>(l1PFCandidates, i));
  }

  //get the jets
  std::vector<edm::Ptr<reco::CaloJet>> jets;
  jets.reserve((*l1PFJets).size());
  for (unsigned int i = 0; i < (*l1PFJets).size(); i++) {
    jets.push_back(edm::Ptr<reco::CaloJet>(l1PFJets, i));
    //
  }

  std::vector<l1t::PFTau> taus;

  taus = processEvent_HW(particles, jets);

  std::sort(taus.begin(), taus.end(), [](const l1t::PFTau& i, const l1t::PFTau& j) { return (i.pt() > j.pt()); });

  iEvent.emplace(tauToken_, std::move(taus));
}

std::vector<l1t::PFTau> L1HPSPFTauProducer::processEvent_HW(std::vector<edm::Ptr<l1t::PFCandidate>>& work,
                                                            std::vector<edm::Ptr<reco::CaloJet>>& jwork) const {
  //convert and call emulator

  using namespace l1HPSPFTauEmu;

  std::vector<Particle> particles = convertEDMToHW(work);

  std::vector<Particle> jets = convertJetsToHW(jwork);
  //also need to pass the jet enabler

  bool jEnable = fUseJets_;

  std::vector<Tau> taus = emulateEvent(particles, jets, jEnable);

  return convertHWToEDM(taus);
}

std::vector<l1HPSPFTauEmu::Particle> L1HPSPFTauProducer::convertJetsToHW(std::vector<edm::Ptr<reco::CaloJet>>& edmJets) {
  using namespace l1HPSPFTauEmu;
  std::vector<Particle> hwJets;
  std::for_each(edmJets.begin(), edmJets.end(), [&](edm::Ptr<reco::CaloJet>& edmJet) {
    l1HPSPFTauEmu::Particle jPart;
    jPart.hwPt = l1ct::Scales::makePtFromFloat(edmJet->pt());
    jPart.hwEta = edmJet->eta() * etaphi_base;
    jPart.hwPhi = edmJet->phi() * etaphi_base;
    jPart.tempZ0 = 0.;
    hwJets.push_back(jPart);
  });
  return hwJets;
}

//conversion to and from HW bitwise
std::vector<l1HPSPFTauEmu::Particle> L1HPSPFTauProducer::convertEDMToHW(
    std::vector<edm::Ptr<l1t::PFCandidate>>& edmParticles) {
  using namespace l1HPSPFTauEmu;
  std::vector<Particle> hwParticles;

  std::for_each(edmParticles.begin(), edmParticles.end(), [&](edm::Ptr<l1t::PFCandidate>& edmParticle) {
    Particle hwPart;
    hwPart.hwPt = l1ct::Scales::makePtFromFloat(edmParticle->pt());
    hwPart.hwEta = edmParticle->eta() * etaphi_base;
    hwPart.hwPhi = edmParticle->phi() * etaphi_base;
    hwPart.pID = edmParticle->id();
    if (edmParticle->z0()) {
      hwPart.tempZ0 = edmParticle->z0() / dz_base;
    }
    hwParticles.push_back(hwPart);
  });
  return hwParticles;
}

std::vector<l1t::PFTau> L1HPSPFTauProducer::convertHWToEDM(std::vector<l1HPSPFTauEmu::Tau> hwTaus) {
  using namespace l1HPSPFTauEmu;
  std::vector<l1t::PFTau> edmTaus;

  //empty array for the PFTau format, since it's used for PuppiTaus but not here
  float tauArray[80] = {0};
  std::for_each(hwTaus.begin(), hwTaus.end(), [&](Tau tau) {
    l1gt::Tau gtTau = tau.toGT();
    l1gt::PackedTau packTau = gtTau.pack();

    l1t::PFTau pTau(
        reco::Candidate::PolarLorentzVector(
            l1ct::Scales::floatPt(tau.hwPt), float(tau.hwEta) / etaphi_base, float(tau.hwPhi) / etaphi_base, 0),
        tauArray,
        0,
        0,
        0,
        tau.hwPt,
        tau.hwEta,
        tau.hwPhi);
    pTau.set_encodedTau(packTau);
    edmTaus.push_back(pTau);
  });
  return edmTaus;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1HPSPFTauProducer);
