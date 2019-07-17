#ifndef L1TRIGGER_PHASE2L1TAUS_TAUNNID_H
#define L1TRIGGER_PHASE2L1TAUS_TAuNNID_H

#include <string>
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "DataFormats/L1Trigger/interface/L1PFTau.h"
#include "DataFormats/Phase2L1ParticleFlow/interface/PFCandidate.h"

class TauNNId  {
    public:
      TauNNId();
      ~TauNNId();
      
      void initialize(std::string iName, const std::string iWeightFile,int iNParticles);
      void SetNNVectorVar();
      float EvaluateNN();
      float compute(l1t::PFCandidate &iSeed,l1t::PFCandidateCollection &iParts);    

      std::string fInput;
      int fNParticles;
      float *fPt;
      float *fEta;
      float *fPhi;
      float *fId;

    private:
      tensorflow::Session* session;
      tensorflow::GraphDef* graphDef;
      std::vector<float> NNvectorVar_; 
};
#endif
