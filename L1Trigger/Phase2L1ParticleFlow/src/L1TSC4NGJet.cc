#include "L1Trigger/Phase2L1ParticleFlow/interface/L1TSC4NGJetID.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include <cmath>

L1TSC4NGJetID::L1TSC4NGJetID(const std::shared_ptr<hls4mlEmulator::Model> model, int iNParticles, bool debug)
    : modelRef_(model) {
  NNvectorVar_.clear();
  fNParticles_ = iNParticles;
  isDebugEnabled_ = debug;

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

void L1TSC4NGJetID::setNNVectorVar() {
  NNvectorVar_.clear();
  if (isDebugEnabled_) {
    LogDebug("L1TSC4NGJetID") << "\n ===== Input Vector =====" << std::endl;
  }

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

    if (isDebugEnabled_) {
      LogDebug("L1TSC4NGJetID") << "Particle: " << i0 << "\n"
                                << "pT: " << NNvectorVar_[i0 * 20]
                                << " | "
                                   "pT rel: "
                                << NNvectorVar_[i0 * 20 + 1]
                                << " | "
                                   "pT log: "
                                << NNvectorVar_[i0 * 20 + 2]
                                << " | "
                                   "dEta: "
                                << NNvectorVar_[i0 * 20 + 3]
                                << " | "
                                   "dPhi: "
                                << NNvectorVar_[i0 * 20 + 4]
                                << " | "
                                   "mass: "
                                << NNvectorVar_[i0 * 20 + 5]
                                << " | "
                                   "photon ID: "
                                << NNvectorVar_[i0 * 20 + 6]
                                << " | "
                                   "electron + ID: "
                                << NNvectorVar_[i0 * 20 + 7]
                                << " | "
                                   "electron - ID: "
                                << NNvectorVar_[i0 * 20 + 8]
                                << " | "
                                   "muon + ID: "
                                << NNvectorVar_[i0 * 20 + 9]
                                << " | "
                                   "muon - ID: "
                                << NNvectorVar_[i0 * 20 + 10]
                                << " | "
                                   "neutral hadron ID: "
                                << NNvectorVar_[i0 * 20 + 11]
                                << " | "
                                   "hadron + ID: "
                                << NNvectorVar_[i0 * 20 + 12]
                                << " | "
                                   "hadron - ID: "
                                << NNvectorVar_[i0 * 20 + 13]
                                << " | "
                                   "z0: "
                                << NNvectorVar_[i0 * 20 + 14]
                                << " | "
                                   "sqrt Dxy: "
                                << NNvectorVar_[i0 * 20 + 15]
                                << " | "
                                   "is filled: "
                                << NNvectorVar_[i0 * 20 + 16]
                                << " | "
                                   "puppi weight: "
                                << NNvectorVar_[i0 * 20 + 17]
                                << " | "
                                   "ElectroMagnetic ID: "
                                << NNvectorVar_[i0 * 20 + 18]
                                << " | "
                                   "Track Quality: "
                                << NNvectorVar_[i0 * 20 + 19] << " | "
                                << "===========" << std::endl;
    }
  }
}

std::vector<float> L1TSC4NGJetID::EvaluateNNFixed() {
  const int NInputs = 320;
  classtype classresult;
  regressiontype regressionresult;

  inputtype fillzero = 0.0;

  inputtype modelInput[NInputs] = {};  // Do something
  std::fill(modelInput, modelInput + NInputs, fillzero);

  for (unsigned int i = 0; i < NNvectorVar_.size(); i++) {
    modelInput[i] = NNvectorVar_[i];
  }

  pairtype modelResult;

  modelRef_->prepare_input(modelInput);
  modelRef_->predict();
  modelRef_->read_result(&modelResult);

  std::vector<float> modelResult_;
  if (isDebugEnabled_) {
    LogDebug("L1TSC4NGJetID") << "\n ===== Jet ID Output Score =====" << std::endl;
  }
  for (unsigned int i = 0; i < 8; i++) {
    modelResult_.push_back(modelResult.second[i].to_float());
    if (isDebugEnabled_) {
      LogDebug("L1TSC4NGJetID") << l1ct::JetTagClassHandler::tagClassesDefault_[i] << " : " << modelResult_[i]
                                << std::endl;
    }
  }
  modelResult_.push_back(modelResult.first[0].to_float());
  if (isDebugEnabled_) {
    LogDebug("L1TSC4NGJetID") << "\n ===== Jet pT Correction Output ===== \n"
                              << modelResult.first[0].to_float() << std::endl;
  }
  return modelResult_;
}  //end EvaluateNNFixed

std::vector<float> L1TSC4NGJetID::computeFixed(const l1t::PFJet &iJet, bool useRawPt) {
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
  float jet_pt_ = float(ctJet.hwPt);
  float jet_eta_ = float(ctJet.hwEta);
  float jet_phi_ = float(ctJet.hwPhi);

  for (unsigned int i0 = 0; i0 < iParts.size(); i0++) {
    if (i0 >= (unsigned int)fNParticles_)
      break;
    fPt_.get()[i0] = iParts[i0]->hwPt();
    fPt_rel_.get()[i0] = iParts[i0]->hwPt() / jet_pt_;

    L1SCJetEmu::detaphi_t dphi(iParts[i0]->hwPhi() - jet_phi_);
    // phi wrap
    L1SCJetEmu::detaphi_t dphi0 = dphi > L1SCJetEmu::detaphi_t(l1ct::Scales::INTPHI_PI)
                                      ? L1SCJetEmu::detaphi_t(l1ct::Scales::INTPHI_TWOPI - dphi)
                                      : L1SCJetEmu::detaphi_t(dphi);
    L1SCJetEmu::detaphi_t dphi1 = dphi < L1SCJetEmu::detaphi_t(-l1ct::Scales::INTPHI_PI)
                                      ? L1SCJetEmu::detaphi_t(l1ct::Scales::INTPHI_TWOPI + dphi)
                                      : L1SCJetEmu::detaphi_t(dphi);
    L1SCJetEmu::detaphi_t dphiw = dphi > L1SCJetEmu::detaphi_t(0) ? dphi0 : dphi1;

    fDEta_.get()[i0] = jet_eta_ - float(iParts[i0]->hwEta());
    fDPhi_.get()[i0] = dphiw;

    fPt_log_.get()[i0] = std::log(iParts[i0]->hwPt());

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
    fZ0_.get()[i0] = iParts[i0]->hwZ0();
    fDxy_.get()[i0] = iParts[i0]->hwDxy();
    fIs_filled_.get()[i0] = 1;
    fPuppi_weight_.get()[i0] = iParts[i0]->hwPuppiWeight();
    fEmID_.get()[i0] = iParts[i0]->hwEmID();
    fQuality_.get()[i0] = iParts[i0]->hwTkQuality();

    fCharge_.get()[i0] = iParts[i0]->charge();
    fId_.get()[i0] = iParts[i0]->id();
  }
  setNNVectorVar();
  return EvaluateNNFixed();
}
