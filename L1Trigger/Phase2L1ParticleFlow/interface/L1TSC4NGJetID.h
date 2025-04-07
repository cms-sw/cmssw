#ifndef L1TRIGGER_PHASE2L1PARTICLEFLOWS_L1TSC4NGJetID_H
#define L1TRIGGER_PHASE2L1PARTICLEFLOWS_L1TSC4NGJetID_H

#include <string>
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "DataFormats/L1TParticleFlow/interface/PFJet.h"
#include "DataFormats/L1TParticleFlow/interface/datatypes.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/jetmet/L1SeedConePFJetEmulator.h"

//HLS4ML compiled emulator modeling
#include <string>
#include "ap_fixed.h"
#include "hls4ml/emulator.h"

class L1TSC4NGJetID {
public:
  L1TSC4NGJetID(const std::shared_ptr<hls4mlEmulator::Model> model, int iNParticles);
  ~L1TSC4NGJetID() = default;

  typedef ap_fixed<24, 12, AP_RND, AP_SAT, 0> inputtype;
  typedef std::array<ap_ufixed<24, 12, AP_RND, AP_SAT, 0>, 8> classtype;
  typedef std::array<ap_fixed<16, 6>, 1> regressiontype;
  typedef std::pair<regressiontype, classtype> pairtype;

  void setNNVectorVar();
  std::vector<float> EvaluateNNFixed();
  std::vector<float> computeFixed(const l1t::PFJet &iJet, bool useRawPt);

private:
  std::vector<inputtype> NNvectorVar_;
  int fNParticles_;
  unique_ptr<float[]> fPt_;
  unique_ptr<float[]> fPt_rel_;
  unique_ptr<float[]> fDEta_;
  unique_ptr<float[]> fDPhi_;
  unique_ptr<float[]> fPt_log_;
  unique_ptr<float[]> fMass_;
  unique_ptr<float[]> fZ0_;
  unique_ptr<float[]> fDxy_;
  unique_ptr<int[]> fIs_filled_;
  unique_ptr<float[]> fPuppi_weight_;
  unique_ptr<int[]> fEmID_;
  unique_ptr<float[]> fQuality_;

  unique_ptr<int[]> fCharge_;
  unique_ptr<int[]> fId_;
  std::shared_ptr<hls4mlEmulator::Model> modelRef_;
};
#endif
