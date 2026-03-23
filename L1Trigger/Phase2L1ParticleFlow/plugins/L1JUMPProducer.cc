#include <vector>
#include <string>
#include <ap_int.h>
#include <ap_fixed.h>
#include <TVector2.h>
#include <iostream>

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "DataFormats/L1TParticleFlow/interface/PFJet.h"

#include "DataFormats/L1TParticleFlow/interface/puppi.h"
#include "DataFormats/L1TParticleFlow/interface/sums.h"
#include "DataFormats/L1TParticleFlow/interface/jets.h"
#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"

#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "L1Trigger/Phase2L1ParticleFlow/interface/jetmet/L1PFJUMPEmulator.h"

class L1JUMPProducer : public edm::global::EDProducer<> {
  /*
    Producer for the JUMP Algorithm
    JUMP: Jet Uncertainty-aware MET Prediction
    - Approximate L1 Jet energy resolution by pT, eta value
    - Apply the estimated resolution to MET
  */
public:
  explicit L1JUMPProducer(const edm::ParameterSet&);
  ~L1JUMPProducer() override;

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
  edm::EDGetTokenT<std::vector<l1t::EtSum>> metToken;
  edm::EDGetTokenT<std::vector<l1t::PFJet>> jetsToken;

  typedef l1ct::pt_t pt_t;
  typedef l1ct::Jet Jet;

  static constexpr float ptLSB_ = 0.25;
  static constexpr float phiLSB_ = M_PI / 720;
  static constexpr float maxPt_ = ((1 << pt_t::width) - 1) * ptLSB_;

  void CalcJUMP_HLS(const l1t::EtSum& metVector,
                    const std::vector<l1ct::Jet>& jets,
                    reco::Candidate::PolarLorentzVector& JUMPVector) const;

  std::vector<l1ct::Jet> convertEDMToHW(const std::vector<l1t::PFJet> edmJets) const;

  double minJetPt;
  double maxJetEta;
  std::string jerFilePath_;
};

L1JUMPProducer::L1JUMPProducer(const edm::ParameterSet& cfg)
    : metToken(consumes<std::vector<l1t::EtSum>>(cfg.getParameter<edm::InputTag>("RawMET"))),
      jetsToken(consumes<std::vector<l1t::PFJet>>(cfg.getParameter<edm::InputTag>("L1PFJets"))),
      minJetPt(cfg.getParameter<double>("MinJetpT")),
      maxJetEta(cfg.getParameter<double>("MaxJetEta")),
      jerFilePath_(cfg.getParameter<std::string>("JERFile")) {
  produces<std::vector<l1t::EtSum>>();
  L1JUMPEmu::SetJERFile(jerFilePath_);
}

void L1JUMPProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // Load Met
  l1t::EtSum rawMET = iEvent.get(metToken)[0];
  // Load Jets
  l1t::PFJetCollection edmJets = iEvent.get(jetsToken);

  std::vector<l1ct::Jet> hwJets = convertEDMToHW(edmJets);  // convert to the emulator format
  // Apply pT and eta selections
  std::vector<l1ct::Jet> hwJetsFiltered;
  std::copy_if(hwJets.begin(), hwJets.end(), std::back_inserter(hwJetsFiltered), [&](auto jet) {
    return jet.hwPt > l1ct::Scales::makePtFromFloat(float(minJetPt)) &&
           std::abs(jet.hwEta) < l1ct::Scales::makeGlbEta(maxJetEta);
  });

  // JUMP Algorithm
  reco::Candidate::PolarLorentzVector JUMPVector;
  CalcJUMP_HLS(rawMET, hwJetsFiltered, JUMPVector);

  l1t::EtSum theJUMP(JUMPVector, l1t::EtSum::EtSumType::kMissingEt, 0, 0, 0, 0);
  auto JUMPCollection = std::make_unique<std::vector<l1t::EtSum>>(0);
  JUMPCollection->push_back(theJUMP);
  iEvent.put(std::move(JUMPCollection));
}

void L1JUMPProducer::CalcJUMP_HLS(const l1t::EtSum& metVector,
                                  const std::vector<l1ct::Jet>& jets,
                                  reco::Candidate::PolarLorentzVector& outMet_Vector) const {
  // JUMP Calculate
  l1ct::Sum inMet;
  inMet.hwPt = metVector.pt();
  inMet.hwPhi = l1ct::Scales::makeGlbPhi(metVector.phi());

  l1ct::Sum outMet;

  JUMP_emu(inMet, jets, outMet);

  outMet_Vector.SetPt(outMet.hwPt.to_double());
  outMet_Vector.SetPhi(outMet.hwPhi.to_double() * phiLSB_);
  outMet_Vector.SetEta(0);
}

std::vector<l1ct::Jet> L1JUMPProducer::convertEDMToHW(const std::vector<l1t::PFJet> edmJets) const {
  std::vector<l1ct::Jet> hwJets;
  std::for_each(edmJets.begin(), edmJets.end(), [&](const l1t::PFJet jet) {
    l1ct::Jet hwJet = l1ct::Jet::unpack(jet.getHWJetCT());
    hwJets.push_back(hwJet);
  });
  return hwJets;
}

L1JUMPProducer::~L1JUMPProducer() {}

DEFINE_FWK_MODULE(L1JUMPProducer);
