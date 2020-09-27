#ifndef L1TRIGGER_PHASE2L1PARTICLEFLOWS_TAUNNID_H
#define L1TRIGGER_PHASE2L1PARTICLEFLOWS_TAUNNID_H

#include <string>
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"

class TauNNId {
public:
  TauNNId();
  ~TauNNId();

  void initialize(const std::string &iName, const std::string &iWeightFile, int iNParticles);
  void SetNNVectorVar();
  float EvaluateNN();
  float compute(const l1t::PFCandidate &iSeed, l1t::PFCandidateCollection &iParts);

  std::string fInput_;
  int fNParticles_;
  unique_ptr<float[]> fPt_;
  unique_ptr<float[]> fEta_;
  unique_ptr<float[]> fPhi_;
  unique_ptr<float[]> fId_;

private:
  tensorflow::Session *session_;
  tensorflow::GraphDef *graphDef_;
  std::vector<float> NNvectorVar_;
};
#endif
