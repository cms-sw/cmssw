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

//Prepare the inputs for the Tau NN
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

// Main architecture of the NN here
Tau_NN_Result TauNNIdHW::EvaluateNN() {
  input_t model_input[N_INPUT_1_1];
  for (unsigned int i = 0; i < NNvectorVar_.size(); i++) {
    model_input[i] = input_t(NNvectorVar_[i]);
  }

  layer2_t layer2_out[N_LAYER_2];
  nnet::dense<input_t, layer2_t, config2>(model_input, layer2_out, w2, b2);  // Dense_1

  layer4_t layer4_out[N_LAYER_2];
  nnet::relu<layer2_t, layer4_t, relu_config4>(layer2_out, layer4_out);  // relu_1

  layer5_t layer5_out[N_LAYER_5];
  nnet::dense<layer4_t, layer5_t, config5>(layer4_out, layer5_out, w5, b5);  // Dense_2

  layer7_t layer7_out[N_LAYER_5];
  nnet::relu<layer5_t, layer7_t, relu_config7>(layer5_out, layer7_out);  // relu_2

  layer8_t layer8_out[N_LAYER_8];
  nnet::dense<layer7_t, layer8_t, config8>(layer7_out, layer8_out, w8, b8);  // Dense_3

  layer10_t layer10_out[N_LAYER_8];
  nnet::relu<layer8_t, layer10_t, relu_config10>(layer8_out, layer10_out);  // relu_3

  layer11_t layer11_out[N_LAYER_11];
  nnet::dense<layer10_t, layer11_t, config11>(layer10_out, layer11_out, w11, b11);  // Dense_4

  layer13_t layer13_out[N_LAYER_11];
  nnet::relu<layer11_t, layer13_t, relu_config13>(layer11_out, layer13_out);  // relu_4

  layer14_t layer14_out[N_LAYER_14];
  nnet::dense<layer13_t, layer14_t, config14>(layer13_out, layer14_out, w14, b14);  // Dense_5

  layer16_t layer16_out[N_LAYER_14];
  nnet::relu<layer14_t, layer16_t, relu_config16>(layer14_out, layer16_out);  // relu_5

  layer17_t layer17_out[N_LAYER_17];
  nnet::dense<layer16_t, layer17_t, config17>(layer16_out, layer17_out, w17, b17);  // Dense_6

  result_t layer19_out[N_LAYER_17];
  nnet::sigmoid<layer17_t, result_t, sigmoid_config19>(layer17_out, layer19_out);  // jetID_output

  result_t layer20_out[N_LAYER_20];
  nnet::dense<layer16_t, result_t, config20>(layer16_out, layer20_out, w20, b20);  // pT_output

  // Return both pT correction and the NN ID
  Tau_NN_Result nn_out;
  nn_out.nn_pt_correction = layer20_out[0];
  nn_out.nn_id = layer19_out[0];

  return nn_out;
}

/*
// Uncomment for debugging purposes
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
