#ifndef L1TRIGGER_PHASE2L1PARTICLEFLOWS_TAUNNID_H
#define L1TRIGGER_PHASE2L1PARTICLEFLOWS_TAUNNID_H

#include <string>
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"

class TauNNId {
public:
  TauNNId(const std::string &iInput,
          const tensorflow::Session *session,
          const std::string &iWeightFile,
          int iNParticles);
  ~TauNNId(){};

  void setNNVectorVar();
  float EvaluateNN();
  float *NNVectorVar() { return NNvectorVar_.data(); }
  float compute(const l1t::PFCandidate &iSeed, l1t::PFCandidateCollection &iParts);

private:
  const tensorflow::Session *session_;
  std::vector<float> NNvectorVar_;
  std::string fInput_;
  int fNParticles_;
  unique_ptr<float[]> fPt_;
  unique_ptr<float[]> fEta_;
  unique_ptr<float[]> fPhi_;
  unique_ptr<int[]> fId_;
};
#endif
