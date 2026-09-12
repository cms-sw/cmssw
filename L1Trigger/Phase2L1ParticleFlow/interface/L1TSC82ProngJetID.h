#ifndef L1TRIGGER_PHASE2L1PARTICLEFLOWS_L1TSC82ProngJetID_H
#define L1TRIGGER_PHASE2L1PARTICLEFLOWS_L1TSC82ProngJetID_H

#include "DataFormats/L1TParticleFlow/interface/PFJet.h"
#include <memory>
#include <vector>

//HLS4ML compiled emulator modeling
#include "ap_fixed.h"
#include "hls4ml/emulator.h"

class L1TSC82ProngJetID {
public:
  L1TSC82ProngJetID(const std::shared_ptr<hls4mlEmulator::Model> model, int iNParticles);

  typedef ap_fixed<24, 12, AP_RND, AP_SAT, 0> inputtype;
  typedef ap_ufixed<20, 10, AP_RND, AP_SAT, 0> prong_score;

  void setNNVectorVar();
  std::vector<float> EvaluateNNFixed();
  std::vector<float> computeFixed(const l1t::PFJet &iJet);

private:
  std::vector<inputtype> NNvectorVar_;
  int fNParticles_;
  std::unique_ptr<float[]> fPt_;
  std::unique_ptr<float[]> fPt_rel_;
  std::unique_ptr<float[]> fDEta_;
  std::unique_ptr<float[]> fDPhi_;
  std::unique_ptr<float[]> fPt_log_;
  std::unique_ptr<float[]> fMass_;
  std::unique_ptr<float[]> fZ0_;
  std::unique_ptr<float[]> fDxy_;
  std::unique_ptr<int[]> fIs_filled_;
  std::unique_ptr<float[]> fPuppi_weight_;
  std::unique_ptr<int[]> fEmID_;
  std::unique_ptr<float[]> fQuality_;

  std::unique_ptr<int[]> fCharge_;
  std::unique_ptr<int[]> fId_;
  std::shared_ptr<hls4mlEmulator::Model> modelRef_;

  //bool isDebugEnabled_;
};
#endif
