#ifndef L1TRIGGER_PHASE2L1PARTICLEFLOWS_BJETID_H
#define L1TRIGGER_PHASE2L1PARTICLEFLOWS_BJETID_H

#include <string>
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "DataFormats/L1TParticleFlow/interface/PFJet.h"

struct BJetTFCache {
  BJetTFCache(const std::string &graphPath) : graphDef(tensorflow::loadGraphDef(graphPath)) {}
  std::unique_ptr<tensorflow::GraphDef> graphDef;
};

class BJetId {
public:
  BJetId(const std::string &iInput,
         const std::string &iOutput,
         const BJetTFCache *cache,
         const std::string &iWeightFile,
         int iNParticles);
  ~BJetId();

  void setNNVectorVar();
  float EvaluateNN();
  float compute(const l1t::PFJet &iJet, float vz, bool useRawPt);

private:
  tensorflow::Session *session_;
  std::vector<float> NNvectorVar_;
  std::string fInput_;
  std::string fOutput_;
  int fNParticles_;
  unique_ptr<float[]> fPt_;
  unique_ptr<float[]> fEta_;
  unique_ptr<float[]> fPhi_;
  unique_ptr<float[]> fId_;
  unique_ptr<int[]> fCharge_;
  unique_ptr<float[]> fDZ_;
  unique_ptr<float[]> fDX_;
  unique_ptr<float[]> fDY_;
};
#endif
