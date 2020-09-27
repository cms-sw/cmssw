#include "L1Trigger/Phase2L1ParticleFlow/interface/TauNNId.h"
#include <iostream>
#include <cmath>
#include <TMath.h>

TauNNId::TauNNId() { NNvectorVar_.clear(); }
TauNNId::~TauNNId() {
  tensorflow::closeSession(session_);
  delete graphDef_;
}
void TauNNId::initialize(const std::string &iInput, const std::string &iWeightFile, int iNParticles) {
  std::string cmssw_base_src = getenv("CMSSW_BASE");
  graphDef_ = tensorflow::loadGraphDef((cmssw_base_src + "/src/" + iWeightFile).c_str());
  session_ = tensorflow::createSession(graphDef_);
  fNParticles_ = iNParticles;
  fPt_ = std::make_unique<float>(fNParticles_);
  fEta_ = std::make_unique<float>(fNParticles_);
  fPhi_ = std::make_unique<float>(fNParticles_);
  fId_ = std::make_unique<float>(fNParticles_);
  fInput_ = iInput;  //tensorflow::run(session_, { { "input_1:0",input } }, { "dense_4/Sigmoid:0" }, &outputs);
}
void TauNNId::SetNNVectorVar() {
  NNvectorVar_.clear();
  for (int i0 = 0; i0 < fNParticles_; i0++) {
    NNvectorVar_.push_back(fPt_.get()[i0]);   //pT
    NNvectorVar_.push_back(fEta_.get()[i0]);  //dEta from jet axis
    NNvectorVar_.push_back(fPhi_.get()[i0]);  //dPhi from jet axis
    if (fPt_.get()[i0] == 0) {
      for (int i1 = 0; i1 < 5; i1++)
        NNvectorVar_.push_back(0);
      continue;
    }
    fId_.get()[i0] == l1t::PFCandidate::Photon ? NNvectorVar_.push_back(1) : NNvectorVar_.push_back(0);    //Photon
    fId_.get()[i0] == l1t::PFCandidate::Electron ? NNvectorVar_.push_back(1) : NNvectorVar_.push_back(0);  //Electron
    fId_.get()[i0] == l1t::PFCandidate::Muon ? NNvectorVar_.push_back(1) : NNvectorVar_.push_back(0);      //Muon
    fId_.get()[i0] == l1t::PFCandidate::NeutralHadron ? NNvectorVar_.push_back(1)
                                                      : NNvectorVar_.push_back(0);  //Neutral Had
    fId_.get()[i0] == l1t::PFCandidate::ChargedHadron ? NNvectorVar_.push_back(1)
                                                      : NNvectorVar_.push_back(0);  //Charged Had
  }
}
float TauNNId::EvaluateNN() {
  tensorflow::Tensor input(tensorflow::DT_FLOAT,
                           {1, (unsigned int)NNvectorVar_.size()});  //was {1,35} but get size mismatch, CHECK
  for (unsigned int i = 0; i < NNvectorVar_.size(); i++) {
    input.matrix<float>()(0, i) = float(NNvectorVar_[i]);
  }
  std::vector<tensorflow::Tensor> outputs;
  tensorflow::run(session_, {{fInput_, input}}, {"dense_4/Sigmoid:0"}, &outputs);
  float disc = outputs[0].matrix<float>()(0, 0);
  return disc;
}  //end EvaluateNN

float TauNNId::compute(l1t::PFCandidate &iSeed, l1t::PFCandidateCollection &iParts) {
  for (int i0 = 0; i0 < fNParticles_; i0++) {
    fPt_.get()[i0] = 0;
    fEta_.get()[i0] = 0;
    fPhi_.get()[i0] = 0;
    fId_.get()[i0] = 0;
  }
  std::sort(iParts.begin(), iParts.end(), [](l1t::PFCandidate i, l1t::PFCandidate j) { return (i.pt() > j.pt()); });
  for (unsigned int i0 = 0; i0 < iParts.size(); i0++) {
    if (i0 > 10)
      break;
    fPt_.get()[i0] = iParts[i0].pt();
    fEta_.get()[i0] = iSeed.eta() - iParts[i0].eta();
    float lDPhi = iSeed.phi() - iParts[i0].phi();
    if (lDPhi > TMath::Pi())
      lDPhi -= TMath::Pi();
    if (lDPhi < -TMath::Pi())
      lDPhi += TMath::Pi();
    fPhi_.get()[i0] = lDPhi;
    fId_.get()[i0] = iParts[i0].id();
  }
  SetNNVectorVar();
  return EvaluateNN();
}
