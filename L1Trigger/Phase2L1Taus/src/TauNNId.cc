#include "L1Trigger/Phase2L1Taus/interface/TauNNId.h"
#include <iostream>
#include <cmath>
#include <TMath.h>

TauNNId::TauNNId(){
  NNvectorVar_.clear();
}
TauNNId::~TauNNId() {
  tensorflow::closeSession(session);
  delete graphDef;
}
void TauNNId::initialize(std::string iInput, const std::string iWeightFile, int iNParticles){
  std::string cmssw_base_src = getenv("CMSSW_BASE");
  graphDef= tensorflow::loadGraphDef((cmssw_base_src+"/src/"+iWeightFile).c_str());
  session = tensorflow::createSession(graphDef);
  fNParticles = iNParticles; 
  fPt  = new float[fNParticles]; 
  fEta = new float[fNParticles]; 
  fPhi = new float[fNParticles]; 
  fId  = new float[fNParticles]; 
  fInput = iInput;    //tensorflow::run(session, { { "input_1:0",input } }, { "dense_4/Sigmoid:0" }, &outputs);
}
void TauNNId::SetNNVectorVar(){
    NNvectorVar_.clear();
    for(int i0 = 0; i0 < fNParticles; i0++) { 
      NNvectorVar_.push_back(fPt[i0]) ;  //pT
      NNvectorVar_.push_back(fEta[i0]) ; //dEta from jet axis
      NNvectorVar_.push_back(fPhi[i0]) ; //dPhi from jet axis
      if(fPt[i0] == 0) {
	for(int i1 = 0; i1 < 5; i1++) NNvectorVar_.push_back(0); 
	continue;
      }
      fId[i0] == l1t::PFCandidate::Photon         ? NNvectorVar_.push_back(1) : NNvectorVar_.push_back(0); //Photon
      fId[i0] == l1t::PFCandidate::Electron       ? NNvectorVar_.push_back(1) : NNvectorVar_.push_back(0); //Electron
      fId[i0] == l1t::PFCandidate::Muon           ? NNvectorVar_.push_back(1) : NNvectorVar_.push_back(0); //Muon
      fId[i0] == l1t::PFCandidate::NeutralHadron  ? NNvectorVar_.push_back(1) : NNvectorVar_.push_back(0); //Neutral Had
      fId[i0] == l1t::PFCandidate::ChargedHadron  ? NNvectorVar_.push_back(1) : NNvectorVar_.push_back(0); //Charged Had
    }
}
float TauNNId::EvaluateNN(){
    tensorflow::Tensor input(tensorflow::DT_FLOAT, {1,(unsigned int)NNvectorVar_.size()});//was {1,35} but get size mismatch, CHECK
    for (unsigned int i = 0; i < NNvectorVar_.size(); i++){
      //std::cout<<"i:"<<i<<" x:"<<NNvectorVar_[i]<<std::endl;
      input.matrix<float>()(0,i) =  float(NNvectorVar_[i]);
    }
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::run(session, { { fInput,input } }, { "dense_4/Sigmoid:0" }, &outputs);
    //std::cout << "===> result " << outputs[0].matrix<float>()(0, 0) << std::endl;
    float disc = outputs[0].matrix<float>()(0, 0);
    return disc;
}//end EvaluateNN

float TauNNId::compute(l1t::PFCandidate &iSeed,l1t::PFCandidateCollection &iParts) {
  for(int i0 = 0; i0 < fNParticles; i0++) {
    fPt[i0]  = 0; 
    fEta[i0] = 0;
    fPhi[i0] = 0; 
    fId [i0] = 0;
  }
  std::sort(iParts.begin(), iParts.end(), [](l1t::PFCandidate i,l1t::PFCandidate j){return(i.pt() > j.pt());});   
  for(unsigned int i0 = 0; i0 < iParts.size(); i0++) { 
    if(i0 > 10) break;
    //std::cout << "===> " << i0 << " -- " << iParts[i0].pt() << std::endl;
    fPt[i0]  = iParts[i0].pt();
    fEta[i0] = iSeed.eta()-iParts[i0].eta();
    float lDPhi = iSeed.phi()-iParts[i0].phi();
    if(lDPhi >  TMath::Pi()) lDPhi-=TMath::Pi();
    if(lDPhi < -TMath::Pi()) lDPhi+=TMath::Pi();
    fPhi[i0] = lDPhi;
    fId[i0]  = iParts[i0].id();
  }
  SetNNVectorVar();
  return EvaluateNN();
}
