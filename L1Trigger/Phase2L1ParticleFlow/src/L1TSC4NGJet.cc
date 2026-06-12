#include "L1Trigger/Phase2L1ParticleFlow/interface/L1TSC4NGJetID.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/common/inversion.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/common/log.h"
#include <algorithm>

using namespace L1TSC4NGJet;

  struct ModelInputs {
        inputtype* candidate_inputs;
        inputtype* jet_inputs;
        int total_candidate_inputs;
        int total_jet_inputs;
  };

  struct ModelOutputs {
        inputtype* jet_class_output;
        inputtype* jet_regression_output;
  };

L1TSC4NGJetID::L1TSC4NGJetID(const std::shared_ptr<hls4mlEmulator::Model> model, int iNParticles, bool debug)
    : modelRef_(model) {
  candidate_vector_.clear();
  fNParticles_ = iNParticles;
  isDebugEnabled_ = debug;

  fPt_ = std::make_unique<inputtype[]>(fNParticles_);
  fPt_rel_ = std::make_unique<inputtype[]>(fNParticles_);
  fDEta_ = std::make_unique<inputtype[]>(fNParticles_);
  fDPhi_ = std::make_unique<inputtype[]>(fNParticles_);
  fPt_log_ = std::make_unique<inputtype[]>(fNParticles_);
  fMass_ = std::make_unique<inputtype[]>(fNParticles_);
  fZ0_ = std::make_unique<inputtype[]>(fNParticles_);
  fDxy_ = std::make_unique<inputtype[]>(fNParticles_);
  fIs_filled_ = std::make_unique<inputtype[]>(fNParticles_);
  fPuppi_weight_ = std::make_unique<inputtype[]>(fNParticles_);
  fEmID_ = std::make_unique<inputtype[]>(fNParticles_);
  fQuality_ = std::make_unique<inputtype[]>(fNParticles_);
  fEta_ = std::make_unique<inputtype[]>(fNParticles_);

  fId_ = std::make_unique<inputtype[]>(fNParticles_);
  fCharge_ = std::make_unique<inputtype[]>(fNParticles_);

  fJetPt_ = 0;
  fJetEta_ = 0;
}

void L1TSC4NGJetID::setVectors() {
  candidate_vector_.clear();
  jet_vector_.clear();
  if (isDebugEnabled_) {
    LogDebug("L1TSC4NGJetID") << "\n ===== Input Vector =====" << std::endl;
  }

  for (int i0 = 0; i0 < fNParticles_; i0++) {
    bool filled = fIs_filled_.get()[i0] == 1;
    inputtype null_value = 0;

    candidate_vector_.push_back(filled ? fPt_.get()[i0] : null_value);      // pt
    candidate_vector_.push_back(filled ? fPt_rel_.get()[i0] : null_value);  // pT as a fraction of jet pT
    candidate_vector_.push_back(filled ? fPt_log_.get()[i0] : null_value);  // pt log
    candidate_vector_.push_back(filled ? fDEta_.get()[i0] : null_value);    // dEta from jet axis
    candidate_vector_.push_back(filled ? fDPhi_.get()[i0] : null_value);    // dPhi from jet axis
    candidate_vector_.push_back(filled ? fMass_.get()[i0] : null_value);    // Mass
    candidate_vector_.push_back(filled ? inputtype(fId_.get()[i0] == l1ct::ParticleID::PHOTON) : null_value);    // Photon
    candidate_vector_.push_back(filled ? inputtype(fId_.get()[i0] == l1ct::ParticleID::ELEPLUS) : null_value);   // Positron
    candidate_vector_.push_back(filled ? inputtype(fId_.get()[i0] == l1ct::ParticleID::ELEMINUS) : null_value);  // Electron
    candidate_vector_.push_back(filled ? inputtype(fId_.get()[i0] == l1ct::ParticleID::MUPLUS) : null_value);    // Anti-muon
    candidate_vector_.push_back(filled ? inputtype(fId_.get()[i0] == l1ct::ParticleID::MUMINUS) : null_value);   // Muon
    candidate_vector_.push_back(filled ? inputtype(fId_.get()[i0] == l1ct::ParticleID::HADZERO) : null_value);
    candidate_vector_.push_back(filled ? inputtype(fId_.get()[i0] == l1ct::ParticleID::HADPLUS) : null_value);   // Anti-Pion
    candidate_vector_.push_back(filled ? inputtype(fId_.get()[i0] == l1ct::ParticleID::HADMINUS) : null_value);  // Pion
    candidate_vector_.push_back(filled ? fZ0_.get()[i0] : null_value);                                           // z0
    candidate_vector_.push_back(filled ? fDxy_.get()[i0] : null_value);                                          // dxy
    candidate_vector_.push_back(filled ? fIs_filled_.get()[i0] : null_value);                                    // isfilled
    candidate_vector_.push_back(filled ? fPuppi_weight_.get()[i0] : null_value);  // puppi weight
    candidate_vector_.push_back(filled ? fEmID_.get()[i0] : null_value);          // emID
    candidate_vector_.push_back(filled ? fQuality_.get()[i0] : null_value);       // quality
    candidate_vector_.push_back(filled ? fEta_.get()[i0] : null_value); // eta

    if (isDebugEnabled_) {
      LogDebug("L1TSC4NGJetID") << "Particle: " << i0 << "\n"
                                << "pT: " << candidate_vector_[i0 * N_candidate_features]
                                << " | "
                                   "pT rel: "
                                << candidate_vector_[i0 * N_candidate_features + 1]
                                << " | "
                                   "pT log: "
                                << candidate_vector_[i0 * N_candidate_features + 2]
                                << " | "
                                   "dEta: "
                                << candidate_vector_[i0 * N_candidate_features + 3]
                                << " | "
                                   "dPhi: "
                                << candidate_vector_[i0 * N_candidate_features + 4]
                                << " | "
                                   "mass: "
                                << candidate_vector_[i0 * N_candidate_features + 5]
                                << " | "
                                   "photon ID: "
                                << candidate_vector_[i0 * N_candidate_features + 6]
                                << " | "
                                   "electron + ID: "
                                << candidate_vector_[i0 * N_candidate_features + 7]
                                << " | "
                                   "electron - ID: "
                                << candidate_vector_[i0 * N_candidate_features + 8]
                                << " | "
                                   "muon + ID: "
                                << candidate_vector_[i0 * N_candidate_features + 9]
                                << " | "
                                   "muon - ID: "
                                << candidate_vector_[i0 * N_candidate_features + 10]
                                << " | "
                                   "neutral hadron ID: "
                                << candidate_vector_[i0 * N_candidate_features + 11]
                                << " | "
                                   "hadron + ID: "
                                << candidate_vector_[i0 * N_candidate_features + 12]
                                << " | "
                                   "hadron - ID: "
                                << candidate_vector_[i0 * N_candidate_features + 13]
                                << " | "
                                   "z0: "
                                << candidate_vector_[i0 * N_candidate_features + 14]
                                << " | "
                                   "sqrt Dxy: "
                                << candidate_vector_[i0 * N_candidate_features + 15]
                                << " | "
                                   "is filled: "
                                << candidate_vector_[i0 * N_candidate_features + 16]
                                << " | "
                                   "puppi weight: "
                                << candidate_vector_[i0 * N_candidate_features + 17]
                                << " | "
                                   "ElectroMagnetic ID: "
                                << candidate_vector_[i0 * N_candidate_features + 18]
                                << " | "
                                   "Track Quality: "
                                << candidate_vector_[i0 * N_candidate_features + 19]
                                << " | "
				   "Eta: "
				<< candidate_vector_[i0 * N_candidate_features + 20] << " | "
				<< "===========" << std::endl;
    }
  }
  // After the particle loop
  jet_vector_.push_back(fJetPt_);
  jet_vector_.push_back(fJetEta_);
}

L1TSC4NGJetID::outputpairtype L1TSC4NGJetID::EvaluateNNFixed() {

  inputtype fillzero = 0.0;

   // Define Candidate inputs and fill fully with 0s. Allows for case when N_candidate_inputs > candidate_vector_.size(). 
  inputtype modelCandidateInput[N_candidate_inputs] = {};  
  std::fill(modelCandidateInput, modelCandidateInput + N_candidate_inputs, fillzero);

  // Fill the candidate inputs from the pre calculated NNvectorVar
  for (unsigned int i = 0; i < candidate_vector_.size(); i++) {
    modelCandidateInput[i] = candidate_vector_[i];
  }

  // Define Jet inputs and fill fully with 0s.
  inputtype modelJetInput[N_jet_inputs] = {};  
  std::fill(modelJetInput, modelJetInput + N_jet_inputs, fillzero);
  for (unsigned int i = 0; i < jet_vector_.size(); i++) {
    modelJetInput[i] = jet_vector_[i];
  }

  // Define input struct
  ModelInputs modelInputStruct;
  // Load candidate and jet inputs 
  modelInputStruct.candidate_inputs = modelCandidateInput;
  modelInputStruct.jet_inputs = modelJetInput;
  modelInputStruct.total_candidate_inputs = N_candidate_features;
  modelInputStruct.total_jet_inputs = N_jet_inputs;

  // Define output struct
  ModelOutputs modelOutputStruct;
  
  // Define and load output with zeros ready for replacement from the model
  inputtype modelClassOutput[N_class_outputs] = {}; 
  inputtype modelRegressionOutput[N_regression_outputs] = {};  
  std::fill(modelClassOutput, modelClassOutput + N_class_outputs, fillzero);
  std::fill(modelRegressionOutput, modelRegressionOutput + N_regression_outputs, fillzero);

  // Load class and regression outputs 
  modelOutputStruct.jet_class_output = modelClassOutput;
  modelOutputStruct.jet_regression_output = modelRegressionOutput;

  // Run the inference
  modelRef_->prepare_input(modelInputStruct);
  modelRef_->predict();
  modelRef_->read_result(&modelOutputStruct);

  outputpairtype modelResult_forOutput;
  if (isDebugEnabled_) {
    LogDebug("L1TSC4NGJetID") << "\n ===== Jet ID Output Score =====" << std::endl;
  }
  for (unsigned int i = 0; i < N_class_outputs; i++) {
    // Cast model output to jet tag score datatype
    modelResult_forOutput.second[i] = l1ct::jet_tag_score_t(modelOutputStruct.jet_class_output[i]);
    if (isDebugEnabled_) {
      LogDebug("L1TSC4NGJetID") << l1ct::JetTagClassHandler::tagClassesDefault_[i] << " : " << modelOutputStruct.jet_class_output[i]
                                << " Cast to Jet Class type: " << modelResult_forOutput.second[i] << std::endl;
    }
  }
  
  for (unsigned int i = 0; i < N_regression_outputs; i++) {
      // Cast model output to transient regression score for jet pt multiplication
      modelResult_forOutput.first[i] = modelOutputStruct.jet_regression_output[i];
      if (isDebugEnabled_) {
         LogDebug("L1TSC4NGJetID") << "\n ===== Jet pT Correction Output ===== \n"
                                    << modelOutputStruct.jet_regression_output[i] << " Cast to Jet pT type: " << modelResult_forOutput.first[i]
                                    << std::endl;
      }
   }

  return modelResult_forOutput;
}  //end EvaluateNNFixed

L1TSC4NGJetID::outputpairtype L1TSC4NGJetID::computeFixed(const l1t::PFJet &iJet) {
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
    fEta_.get()[i0] = 0;

    fId_.get()[i0] = 0;
    fCharge_.get()[i0] = 0;
  }
  auto iParts = iJet.constituents();
  // Use stable sort for deterministic ordering when equal pT to match FW
  std::stable_sort(iParts.begin(), iParts.end(), [](edm::Ptr<l1t::PFCandidate> i, edm::Ptr<l1t::PFCandidate> j) {
    return (i->pt() > j->pt());
  });

  l1ct::Jet ctJet = l1ct::Jet::unpack(iJet.getHWJetCT());
  inputtype jet_pt_ = inputtype(ctJet.hwPt);
  inputtype jet_eta_ = inputtype(ctJet.hwEta);
  inputtype jet_phi_ = inputtype(ctJet.hwPhi);
  
  // Fill jet level features
  fJetPt_ = jet_pt_;
  fJetEta_ = jet_eta_;


  for (unsigned int i0 = 0; i0 < iParts.size(); i0++) {
    if (i0 >= (unsigned int)fNParticles_)
      break;
    l1ct::PuppiObj puppicand = l1ct::PuppiObj::unpack(iParts[i0]->encodedPuppi64());
    fPt_.get()[i0] = inputtype(puppicand.hwPt);

    constexpr int INV_LUT_SIZE = 1024;
    inputtype inv_jet_pt = l1ct::invert_with_shift<l1ct::pt_t, l1ct::pt_t, INV_LUT_SIZE>(jet_pt_);

    fPt_rel_.get()[i0] = inputtype(puppicand.hwPt) * inv_jet_pt;

    L1SCJetEmu::detaphi_t dphi(puppicand.hwPhi - jet_phi_);
    // phi wrap
    L1SCJetEmu::detaphi_t dphi0 = dphi > L1SCJetEmu::detaphi_t(l1ct::Scales::INTPHI_PI)
                                      ? L1SCJetEmu::detaphi_t(dphi - l1ct::Scales::INTPHI_TWOPI)
                                      : L1SCJetEmu::detaphi_t(dphi);
    L1SCJetEmu::detaphi_t dphi1 = dphi < L1SCJetEmu::detaphi_t(-l1ct::Scales::INTPHI_PI)
                                      ? L1SCJetEmu::detaphi_t(l1ct::Scales::INTPHI_TWOPI + dphi)
                                      : L1SCJetEmu::detaphi_t(dphi);
    inputtype dphiw = inputtype(dphi > L1SCJetEmu::detaphi_t(0) ? dphi0 : dphi1);

    fDEta_.get()[i0] = jet_eta_ - inputtype(puppicand.hwEta);
    fDPhi_.get()[i0] = dphiw;

    constexpr int LOG_LUT_SIZE = 256;
    inputtype log_pt = l1ct::log_with_shift<l1ct::pt_t, l1ct::pt_t, LOG_LUT_SIZE>(puppicand.hwPt);
    fPt_log_.get()[i0] = log_pt;

    inputtype massCand = L1TSC4NGJet::candidate_mass<inputtype>(puppicand);
    fMass_.get()[i0] = inputtype(massCand);

    inputtype const_eta = inputtype(puppicand.hwEta);
    fEta_.get()[i0] = (const_eta < 0)
      ? inputtype(-const_eta)
      : inputtype(const_eta);

    fZ0_.get()[i0] = puppicand.hwId.charged() ? inputtype(puppicand.hwZ0()) : inputtype(0);
    fDxy_.get()[i0] = puppicand.hwId.charged() ? inputtype(puppicand.hwDxy()) : inputtype(0);
    fIs_filled_.get()[i0] = inputtype(1);
    fPuppi_weight_.get()[i0] = puppicand.hwId.neutral() ? inputtype(puppicand.hwPuppiW()) : inputtype(0);
    fEmID_.get()[i0] = puppicand.hwId.neutral() ? inputtype(puppicand.hwEmID()) : inputtype(0);
    fQuality_.get()[i0] = puppicand.hwId.charged() ? inputtype(puppicand.hwTkQuality()) : inputtype(0);

    fCharge_.get()[i0] = inputtype(puppicand.hwId.charged());
    fId_.get()[i0] = inputtype(puppicand.hwId.bits);
  }
  setVectors();
  return EvaluateNNFixed();
}
