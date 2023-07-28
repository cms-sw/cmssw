#include <iostream>
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/TauNNIdHW.h"

TauNNIdHW::TauNNIdHW() { NNvectorVar_.clear(); }
TauNNIdHW::~TauNNIdHW() {}

void TauNNIdHW::initialize(const std::string &iInput, int iNParticles) {
  fNParticles_ = iNParticles;
  fPt_ = std::make_unique<pt_t[]>(fNParticles_);
  fEta_ = std::make_unique<etaphi_t[]>(fNParticles_);
  fPhi_ = std::make_unique<etaphi_t[]>(fNParticles_);
  fId_ = std::make_unique<id_t[]>(fNParticles_);
  fInput_ = iInput;
}
void TauNNIdHW::SetNNVectorVar() {
  NNvectorVar_.clear();
  for (unsigned i0 = 0; i0 < fNParticles_; i0++) {
    input_t pPt = input_t(fPt_.get()[i0]);
    input_t pEta = input_t(fEta_.get()[i0]);
    input_t pPhi = input_t(fPhi_.get()[i0]);

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

result_t TauNNIdHW::EvaluateNN() {
  input_t data[N_INPUTS];
  for (unsigned int i = 0; i < NNvectorVar_.size(); i++) {
    data[i] = input_t(NNvectorVar_[i]);
  }

  layer1_t layer1_out[N_LAYER_1];
  layer1_t logits1[N_LAYER_1];
  nnet::compute_layer<input_t, layer1_t, config1>(data, logits1, w1, b1);
  nnet::relu<layer1_t, layer1_t, relu_config1>(logits1, layer1_out);

  layer2_t layer2_out[N_LAYER_2];
  layer2_t logits2[N_LAYER_2];
  nnet::compute_layer<layer1_t, layer2_t, config2>(layer1_out, logits2, w2, b2);
  nnet::relu<layer2_t, layer2_t, relu_config2>(logits2, layer2_out);

  layer3_t layer3_out[N_LAYER_3];
  layer3_t logits3[N_LAYER_3];
  nnet::compute_layer<layer2_t, layer3_t, config3>(layer2_out, logits3, w3, b3);
  nnet::relu<layer3_t, layer3_t, relu_config3>(logits3, layer3_out);

  result_t logits4[N_OUTPUTS];
  nnet::compute_layer<layer3_t, result_t, config4>(layer3_out, logits4, w4, b4);
  result_t res[N_OUTPUTS];
  nnet::sigmoid<result_t, result_t, sigmoid_config4>(logits4, res);

  return res[0];
}
/*
void TauNNIdHW::print() { 
  for (unsigned i0 = 0; i0 < fNParticles_; i0++) {
    input_t pPt  = input_t(fPt_.get()[i0]);
    input_t pEta = input_t(fEta_.get()[i0]);
    input_t pPhi = input_t(fPhi_.get()[i0]);
    input_t pId  = input_t(fId_.get()[i0]);    
    fprintf(file_, " %08x", pPt.to_uint());
    fprintf(file_, " %08x", pEta.to_uint());
    fprintf(file_, " %08x", pPhi.to_uint());
    fprintf(file_, " %08x", pId.to_uint());
  }
  fprintf(file_, "\n");
}
*/
result_t TauNNIdHW::compute(const l1t::PFCandidate &iSeed, std::vector<l1t::PFCandidate> &iParts) {
  for (unsigned i0 = 0; i0 < fNParticles_; i0++) {
    fPt_.get()[i0] = 0.;
    fEta_.get()[i0] = 0.;
    fPhi_.get()[i0] = 0.;
    fId_.get()[i0] = 0.;
  }
  std::sort(iParts.begin(), iParts.end(), [](l1t::PFCandidate i, l1t::PFCandidate j) {
    return (pt_t(i.pt()) > pt_t(j.pt()));
  });
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
  SetNNVectorVar();
  return EvaluateNN();
}
