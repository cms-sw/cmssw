#include "L1Trigger/Phase2L1ParticleFlow/interface/L1TSC82ProngJetID.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include <cmath>

L1TSC82ProngJetID::L1TSC82ProngJetID(const std::shared_ptr<hls4mlEmulator::Model> model, int iNParticles)
    : modelRef_(model) {
  NNvectorVar_.clear();
  fNParticles_ = iNParticles;

  fPt_ = std::make_unique<float[]>(fNParticles_);
  fPt_rel_ = std::make_unique<float[]>(fNParticles_);
  fDEta_ = std::make_unique<float[]>(fNParticles_);
  fDPhi_ = std::make_unique<float[]>(fNParticles_);
  fPt_log_ = std::make_unique<float[]>(fNParticles_);
  fMass_ = std::make_unique<float[]>(fNParticles_);
  fZ0_ = std::make_unique<float[]>(fNParticles_);
  fDxy_ = std::make_unique<float[]>(fNParticles_);
  fIs_filled_ = std::make_unique<int[]>(fNParticles_);
  fPuppi_weight_ = std::make_unique<float[]>(fNParticles_);
  fEmID_ = std::make_unique<int[]>(fNParticles_);
  fQuality_ = std::make_unique<float[]>(fNParticles_);

  fId_ = std::make_unique<int[]>(fNParticles_);
  fCharge_ = std::make_unique<int[]>(fNParticles_);
}

void L1TSC82ProngJetID::setNNVectorVar() {
  NNvectorVar_.clear();
  for (int i0 = 0; i0 < fNParticles_; i0++) {
    NNvectorVar_.push_back(fPt_.get()[i0]);                              // pt
    NNvectorVar_.push_back(fPt_rel_.get()[i0]);                          //pT as a fraction of jet pT
    NNvectorVar_.push_back(fPt_log_.get()[i0]);                          // pt log
    NNvectorVar_.push_back(fDEta_.get()[i0]);                            //dEta from jet axis
    NNvectorVar_.push_back(fDPhi_.get()[i0]);                            //dPhi from jet axis
    NNvectorVar_.push_back(fMass_.get()[i0]);                            // Mass
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::Photon);  // Photon
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::Electron && fCharge_.get()[i0] > 0);       // Positron
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::Electron && fCharge_.get()[i0] < 0);       // Electron
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::Muon && fCharge_.get()[i0] > 0);           // Anti-muon
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::Muon && fCharge_.get()[i0] < 0);           // Muon
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::NeutralHadron);                            // Neutral Had
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::ChargedHadron && fCharge_.get()[i0] > 0);  // Anti-Pion
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::ChargedHadron && fCharge_.get()[i0] < 0);  // Pion
    NNvectorVar_.push_back(fZ0_.get()[i0]);                                                               // z0
    NNvectorVar_.push_back(fDxy_.get()[i0]);                                                              // dxy
    NNvectorVar_.push_back(fIs_filled_.get()[i0]);                                                        // isfilled
    NNvectorVar_.push_back(fPuppi_weight_.get()[i0]);  // puppi weight
    NNvectorVar_.push_back(fEmID_.get()[i0]);          // emID
    NNvectorVar_.push_back(fQuality_.get()[i0]);       // quality
  }
}

std::vector<float> L1TSC82ProngJetID::EvaluateNNFixed() {
  const int NInputs = 160;
  prong_score prong_scores;
  inputtype fillzero = 0.0;

  inputtype modelInput[NInputs] = {};  // Do something
  std::fill(modelInput, modelInput + NInputs, fillzero);

  for (unsigned int i = 0; i < NNvectorVar_.size(); i++) {
    modelInput[i] = NNvectorVar_[i];
  }

  modelRef_->prepare_input(modelInput);
  modelRef_->predict();
  modelRef_->read_result(&prong_scores);

  std::vector<float> prong_score_;
  prong_score_.push_back(prong_scores.to_float());

  return prong_score_;
}  //end EvaluateNNFixed

std::vector<float> L1TSC82ProngJetID::computeFixed(const l1t::PFJet &iJet) {
  for (int i0 = 0; i0 < fNParticles_; i0++) {
    fPt_rel_.get()[i0] = 0;
    fPt_.get()[i0] = 0;
    fDEta_.get()[i0] = 0;
    fDPhi_.get()[i0] = 0;
    fPt_log_.get()[i0] = 0;
    fMass_.get()[i0] = 0;
    fZ0_.get()[i0] = 0;
    fDxy_.get()[i0] = 0;
    fIs_filled_.get()[i0] = 0;
    fPuppi_weight_.get()[i0] = 0;
    fEmID_.get()[i0] = 0;
    fQuality_.get()[i0] = 0;

    fId_.get()[i0] = 0;
    fCharge_.get()[i0] = 0;
  }
  auto iParts = iJet.constituents();
  std::sort(iParts.begin(), iParts.end(), [](edm::Ptr<l1t::PFCandidate> i, edm::Ptr<l1t::PFCandidate> j) {
    return (i->pt() > j->pt());
  });

  l1ct::Jet ctJet = l1ct::Jet::unpack(iJet.getHWJetCT());
  float jet_pt_ = ctJet.floatPt();
  float jet_eta_ = ctJet.floatEta();
  float jet_phi_ = ctJet.floatPhi();

  for (unsigned int i0 = 0; i0 < iParts.size(); i0++) {
    if (i0 >= (unsigned int)fNParticles_)
      break;
    fPt_.get()[i0] = iParts[i0]->pt();
    fZ0_.get()[i0] = iParts[i0]->z0();
    fPt_rel_.get()[i0] = iParts[i0]->pt() / jet_pt_;
    fPt_log_.get()[i0] = std::log(iParts[i0]->pt());
    fDEta_.get()[i0] = jet_eta_ - float(iParts[i0]->eta());

    if (1 < iParts[i0]->phi() - jet_phi_) {
      fDPhi_.get()[i0] = 1;
    } else if (-1 > iParts[i0]->phi() - jet_phi_) {
      fDPhi_.get()[i0] = -1;
    } else {
      fDPhi_.get()[i0] = iParts[i0]->phi() - jet_phi_;
    }

    fIs_filled_.get()[i0] = 1;

    float massCand = 0.13f;
    if (abs(iParts[i0]->charge())) {
      if ((iParts[i0]->id() == l1t::PFCandidate::Muon)) {
        massCand = 0.105;
      } else if ((iParts[i0]->id() == l1t::PFCandidate::Electron)) {
        massCand = 0.005;
      }
    } else {
      massCand = iParts[i0]->id() == l1t::PFCandidate::Photon ? 0.0 : 0.5;
    }

    fMass_.get()[i0] = massCand;
    fPuppi_weight_.get()[i0] = iParts[i0]->puppiWeight();
    fId_.get()[i0] = iParts[i0]->id();
    fCharge_.get()[i0] = iParts[i0]->charge();

    fDxy_.get()[i0] = iParts[i0]->hwDxy();
    fEmID_.get()[i0] = iParts[i0]->hwEmID();
    fQuality_.get()[i0] = iParts[i0]->hwTkQuality();
  }
  setNNVectorVar();
  return EvaluateNNFixed();
}
