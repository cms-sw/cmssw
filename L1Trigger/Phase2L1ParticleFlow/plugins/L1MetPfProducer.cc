#include <vector>
#include <string>
#include <ap_int.h>
#include <ap_fixed.h>
#include <TVector2.h>

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
// For HLS MET Data Formats
#include "DataFormats/L1TParticleFlow/interface/puppi.h"
#include "DataFormats/L1TParticleFlow/interface/sums.h"
#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"

#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "L1Trigger/Phase2L1ParticleFlow/interface/jetmet/L1PFMetEmulator.h"

#include "hls4ml/emulator.h"

using namespace l1t;

class L1MetPfProducer : public edm::global::EDProducer<> {
public:
  explicit L1MetPfProducer(const edm::ParameterSet&);
  ~L1MetPfProducer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
  edm::EDGetTokenT<std::vector<l1t::PFCandidate>> _l1PFToken;

  int maxCands_ = 128;

  // quantization controllers
  typedef l1ct::pt_t pt_t;
  typedef l1ct::glbphi_t phi_t;
  static constexpr float phiLSB_ = M_PI / 720;  // rad

  // hls4ml emulator objects
  bool useMlModel_;
  std::shared_ptr<hls4mlEmulator::Model> model;
  std::string modelVersion_;

  typedef ap_fixed<32, 16> input_t;
  typedef ap_fixed<32, 16> result_t;
  static constexpr int numContInputs_ = 4;
  static constexpr int numPxPyInputs_ = 2;
  static constexpr int numCatInputs_ = 2;
  static constexpr int numInputs_ = numContInputs_ + numPxPyInputs_ + numCatInputs_;

  void CalcMetHLS(const std::vector<l1t::PFCandidate>& pfcands, reco::Candidate::PolarLorentzVector& metVector) const;

  int EncodePdgId(int pdgId) const;

  void CalcMlMet(const std::vector<l1t::PFCandidate>& pfcands, reco::Candidate::PolarLorentzVector& metVector) const;
};

L1MetPfProducer::L1MetPfProducer(const edm::ParameterSet& cfg)
    : _l1PFToken(consumes<std::vector<l1t::PFCandidate>>(cfg.getParameter<edm::InputTag>("L1PFObjects"))),
      maxCands_(cfg.getParameter<int>("maxCands")),
      modelVersion_(cfg.getParameter<std::string>("modelVersion")) {
  produces<std::vector<l1t::EtSum>>();
  useMlModel_ = (!modelVersion_.empty());
  if (useMlModel_) {
    hls4mlEmulator::ModelLoader loader(modelVersion_);
    model = loader.load_model();
  } else {
    edm::FileInPath f = cfg.getParameter<edm::FileInPath>("Poly2File");
    L1METEmu::SetPoly2File(f.fullPath());
  }
}

void L1MetPfProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("L1PFObjects", edm::InputTag("L1PFProducer", "l1pfCandidates"));
  desc.add<int>("maxCands", 128);
  desc.add<std::string>("modelVersion", "");
  desc.add<edm::FileInPath>("Poly2File",
                            edm::FileInPath("L1Trigger/Phase2L1ParticleFlow/data/met/l1met_ptphi2pxpy_poly2_v1.json"));
  descriptions.add("L1MetPfProducer", desc);
}

void L1MetPfProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<l1t::PFCandidateCollection> l1PFCandidates;
  iEvent.getByToken(_l1PFToken, l1PFCandidates);

  const std::vector<l1t::PFCandidate>& pfcands = *l1PFCandidates;
  reco::Candidate::PolarLorentzVector metVector;

  if (useMlModel_) {
    CalcMlMet(pfcands, metVector);
  } else {
    CalcMetHLS(pfcands, metVector);
  }

  l1t::EtSum theMET(metVector, l1t::EtSum::EtSumType::kMissingEt, 0, 0, 0, 0);

  auto metCollection = std::make_unique<std::vector<l1t::EtSum>>(0);
  metCollection->push_back(theMET);
  iEvent.put(std::move(metCollection));
}

void L1MetPfProducer::CalcMetHLS(const std::vector<l1t::PFCandidate>& pfcands,
                                 reco::Candidate::PolarLorentzVector& metVector) const {
  std::vector<l1ct::PuppiObjEmu> particles;
  l1ct::Sum hw_met;

  for (int i = 0; i < int(pfcands.size()) && (i < maxCands_ || maxCands_ < 0); i++) {
    const auto& cand = pfcands[i];
    l1ct::PuppiObjEmu each_particle;
    each_particle.initFromBits(cand.encodedPuppi64());
    particles.push_back(each_particle);
  }

  puppimet_emu(particles, hw_met);

  metVector.SetPt(hw_met.hwPt.to_double());
  metVector.SetPhi(hw_met.hwPhi.to_double() * phiLSB_);
  metVector.SetEta(0);
}

int L1MetPfProducer::EncodePdgId(int pdgId) const {
  switch (abs(pdgId)) {
    case 211:  // charged hadron (pion)
      return 1;
    case 130:  // neutral hadron (kaon)
      return 2;
    case 22:  // photon
      return 3;
    case 13:  // muon
      return 4;
    case 11:  // electron
      return 5;
    default:
      return 0;
  }
}

void L1MetPfProducer::CalcMlMet(const std::vector<l1t::PFCandidate>& pfcands,
                                reco::Candidate::PolarLorentzVector& metVector) const {
  std::vector<float> pt;
  std::vector<float> eta;
  std::vector<float> phi;
  std::vector<float> puppiWeight;
  std::vector<int> pdgId;
  std::vector<int> charge;

  for (int i = 0; i < int(pfcands.size()) && (i < maxCands_ || maxCands_ < 0); i++) {
    const auto& l1PFCand = pfcands[i];
    pt.push_back(l1PFCand.pt());
    eta.push_back(l1PFCand.eta());
    phi.push_back(l1PFCand.phi());
    puppiWeight.push_back(l1PFCand.puppiWeight());
    pdgId.push_back(l1PFCand.pdgId());
    charge.push_back(l1PFCand.charge());
  }

  const int inputSize = maxCands_ * numInputs_;

  input_t input[800];
  result_t result[2];

  // initialize with zeros (for padding)
  for (int i = 0; i < inputSize; i++) {
    input[i] = 0;
  }

  for (unsigned int i = 0; i < pt.size(); i++) {
    // input_cont
    input[i * numContInputs_] = pt[i];
    input[i * numContInputs_ + 1] = eta[i];
    input[i * numContInputs_ + 2] = phi[i];
    input[i * numContInputs_ + 3] = puppiWeight[i];
    // input_pxpy
    input[(maxCands_ * numContInputs_) + (i * numPxPyInputs_)] = pt[i] * cos(phi[i]);
    input[(maxCands_ * numContInputs_) + (i * numPxPyInputs_) + 1] = pt[i] * sin(phi[i]);
    // input_cat0
    input[maxCands_ * (numContInputs_ + numPxPyInputs_) + i] = EncodePdgId(pdgId[i]);
    // input_cat1
    input[maxCands_ * (numContInputs_ + numPxPyInputs_ + 1) + i] = (abs(charge[i]) <= 1) ? (charge[i] + 2) : 0;
  }

  model->prepare_input(input);
  model->predict();
  model->read_result(result);

  double met_px = -result[0].to_double();
  double met_py = -result[1].to_double();
  metVector.SetPt(hypot(met_px, met_py));
  metVector.SetPhi(atan2(met_py, met_px));
  metVector.SetEta(0);
}

L1MetPfProducer::~L1MetPfProducer() {}

DEFINE_FWK_MODULE(L1MetPfProducer);
