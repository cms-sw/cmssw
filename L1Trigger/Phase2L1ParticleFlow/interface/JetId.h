#ifndef L1TRIGGER_PHASE2L1PARTICLEFLOWS_JETID_H
#define L1TRIGGER_PHASE2L1PARTICLEFLOWS_JETID_H

#include <string>
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "DataFormats/L1TParticleFlow/interface/PFJet.h"

//HLS4ML compiled emulator modeling
#include <string>
#include "ap_fixed.h"
#include "hls4ml/emulator.h"

struct BJetTFCache {
  BJetTFCache(const std::string &graphPath) : graphDef(tensorflow::loadGraphDef(graphPath)) {
    session = tensorflow::createSession(graphDef.get());
  }
  ~BJetTFCache() { tensorflow::closeSession(session); }
  std::unique_ptr<tensorflow::GraphDef> graphDef;
  tensorflow::Session *session;
};

class JetId {
public:
  JetId(const std::string &iInput,
        const std::string &iOutput,
        const std::shared_ptr<hls4mlEmulator::Model> model,
        int iNParticles);
  JetId(const std::string &iInput, const std::string &iOutput, const BJetTFCache *cache, int iNParticles);
  ~JetId() = default;

  void setNNVectorVar();
  float EvaluateNN();
  ap_fixed<16, 6> EvaluateNNFixed();
  float compute(const l1t::PFJet &iJet, float vz, bool useRawPt);
  ap_fixed<16, 6> computeFixed(const l1t::PFJet &iJet, float vz, bool useRawPt);

private:
  std::vector<float> NNvectorVar_;
  std::string fInput_;
  std::string fOutput_;
  int fNParticles_;
  unique_ptr<float[]> fPt_;
  unique_ptr<float[]> fEta_;
  unique_ptr<float[]> fPhi_;
  unique_ptr<int[]> fId_;
  unique_ptr<int[]> fCharge_;
  unique_ptr<float[]> fDZ_;
  unique_ptr<float[]> fDX_;
  unique_ptr<float[]> fDY_;
  tensorflow::Session *sessionRef_;
  std::shared_ptr<hls4mlEmulator::Model> modelRef_;
};
#endif
