#include <array>
#include <iostream>
#include <utility>
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/TauNNIdHW.h"

TauNNIdHW::TauNNIdHW(const std::shared_ptr<hls4mlEmulator::Model> model) : modelRef_(model) { NNvectorVar_.clear(); }

void TauNNIdHW::initialize(const std::string &iInput, int iNParticles) {
  fNParticles_ = iNParticles;
  fPt_ = std::make_unique<pt_t[]>(fNParticles_);
  fEta_ = std::make_unique<etaphi_t[]>(fNParticles_);
  fPhi_ = std::make_unique<etaphi_t[]>(fNParticles_);
  fId_ = std::make_unique<id_t[]>(fNParticles_);
  fInput_ = iInput;
}

//Prepare the inputs for the Tau NN
void TauNNIdHW::SetNNVectorVar() {
  NNvectorVar_.clear();
  for (unsigned i0 = 0; i0 < fNParticles_; i0++) {
    L1TauEmu::tauinput_t pPt = L1TauEmu::tauinput_t(fPt_.get()[i0]);
    L1TauEmu::tauinput_t pEta = L1TauEmu::tauinput_t(fEta_.get()[i0]);
    L1TauEmu::tauinput_t pPhi = L1TauEmu::tauinput_t(fPhi_.get()[i0]);

    NNvectorVar_.push_back(pPt);
    NNvectorVar_.push_back(pEta);
    NNvectorVar_.push_back(pPhi);
    if (fPt_.get()[i0] == 0) {
      for (unsigned i1 = 0; i1 < 5; i1++)
        NNvectorVar_.push_back(0);
      continue;
    }
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::Photon);         // Photon
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::Electron);       // Electron
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::Muon);           // Muon
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::NeutralHadron);  // Neutral Had
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::ChargedHadron);  // Charged Had
  }
}

// Main architecture of the NN here: delegates to the externally-built
// NNPuppiTauModel package (loaded via hls4mlEmulator::Model) instead of
// in-tree weight arrays, avoiding ELF symbol clashes between hls4ml models
// loaded into the same process (cms-sw/cmssw#49632).
Tau_NN_Result TauNNIdHW::EvaluateNN() {
  constexpr unsigned kNInputs = 80;
  L1TauEmu::tauinput_t model_input[kNInputs];
  for (unsigned int i = 0; i < NNvectorVar_.size(); i++) {
    model_input[i] = L1TauEmu::tauinput_t(NNvectorVar_[i]);
  }

  typedef std::pair<std::array<L1TauEmu::tauresult_t, 1>, std::array<L1TauEmu::tauresult_t, 1>> pairtype;
  pairtype modelResult;
  modelRef_->prepare_input(model_input);
  modelRef_->predict();
  modelRef_->read_result(&modelResult);

  // Return both pT correction and the NN ID
  Tau_NN_Result nn_out;
  nn_out.nn_pt_correction = modelResult.first[0];
  nn_out.nn_id = modelResult.second[0];

  return nn_out;
}

/*
// Uncomment for debugging purposes
void TauNNIdHW::print() { 
  for (unsigned i0 = 0; i0 < fNParticles_; i0++) {
    tauinput_t pPt  = tauinput_t(fPt_.get()[i0]);
    tauinput_t pEta = tauinput_t(fEta_.get()[i0]);
    tauinput_t pPhi = tauinput_t(fPhi_.get()[i0]);
    tauinput_t pId  = tauinput_t(fId_.get()[i0]);
    fprintf(file_, " %08x", pPt.to_uint());
    fprintf(file_, " %08x", pEta.to_uint());
    fprintf(file_, " %08x", pPhi.to_uint());
    fprintf(file_, " %08x", pId.to_uint());
  }
  fprintf(file_, "\n");
}
*/

Tau_NN_Result TauNNIdHW::compute(const l1t::PFCandidate &iSeed, std::vector<l1t::PFCandidate> &iParts) {
  // Initialize the input vector
  for (unsigned i0 = 0; i0 < fNParticles_; i0++) {
    fPt_.get()[i0] = 0.;
    fEta_.get()[i0] = 0.;
    fPhi_.get()[i0] = 0.;
    fId_.get()[i0] = 0.;
  }

  // Sort the candidates by pT
  std::sort(iParts.begin(), iParts.end(), [](l1t::PFCandidate i, l1t::PFCandidate j) {
    return (pt_t(i.pt()) > pt_t(j.pt()));
  });

  // Compute the values w.r.t to the seeds
  for (unsigned int i0 = 0; i0 < iParts.size(); i0++) {
    if (i0 >= fNParticles_)
      break;

    fPt_.get()[i0] = pt_t(iParts[i0].pt());
    fEta_.get()[i0] = etaphi_t(iSeed.eta() - iParts[i0].eta());
    etaphi_t lDPhi = etaphi_t(iSeed.phi()) - etaphi_t(iParts[i0].phi());
    etaphi_t lMPI = 3.1415;

    if (lDPhi > lMPI)
      lDPhi = lDPhi - lMPI;
    if (lDPhi < -lMPI)
      lDPhi = lDPhi + lMPI;

    fPhi_.get()[i0] = lDPhi;
    fId_.get()[i0] = id_t(iParts[i0].id());
  }

  // Set the inputs
  SetNNVectorVar();

  // Return the N outputs with the inputs
  return EvaluateNN();
}
