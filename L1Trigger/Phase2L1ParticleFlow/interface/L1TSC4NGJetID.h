#ifndef L1TRIGGER_PHASE2L1PARTICLEFLOWS_L1TSC4NGJetID_H
#define L1TRIGGER_PHASE2L1PARTICLEFLOWS_L1TSC4NGJetID_H

#include <string>
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "DataFormats/L1TParticleFlow/interface/PFJet.h"
#include "DataFormats/L1TParticleFlow/interface/datatypes.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/jetmet/L1SeedConePFJetEmulator.h"

//HLS4ML compiled emulator modeling
#include "ap_fixed.h"
#include "hls4ml/emulator.h"

namespace L1TSC4NGJet {

  template <class t>
  t candidate_mass(l1ct::PuppiObj puppicand) {
    // Define lookup table
    static const t PION_MASS = t(0.13);
    static const t PHOTON_MASS = t(0.0);
    static const t ELECTRON_MASS = t(0.005);
    static const t MUON_MASS = t(0.105);
    static const t K_MASS = t(0.5);

    // Default to pion mass
    t massCand = PION_MASS;

    if (puppicand.hwId.bits == l1ct::ParticleID::PHOTON) {
      massCand = PHOTON_MASS;
    } else if (puppicand.hwId.bits == l1ct::ParticleID::ELEPLUS || puppicand.hwId.bits == l1ct::ParticleID::ELEMINUS) {
      massCand = ELECTRON_MASS;
    } else if (puppicand.hwId.bits == l1ct::ParticleID::MUMINUS || puppicand.hwId.bits == l1ct::ParticleID::MUPLUS) {
      massCand = MUON_MASS;
    } else if (puppicand.hwId.bits == l1ct::ParticleID::HADZERO) {
      massCand = K_MASS;
    }

    return massCand;
  }

}  // namespace L1TSC4NGJet
class L1TSC4NGJetID {
public:
  L1TSC4NGJetID(const std::shared_ptr<hls4mlEmulator::Model> model, int iNParticles, bool debug);

  typedef ap_fixed<24, 12, AP_RND, AP_SAT, 0> inputtype;
  typedef std::array<ap_ufixed<24, 12, AP_RND, AP_SAT, 0>, 8> classtype;
  typedef std::array<ap_fixed<16, 6>, 1> regressiontype;
  typedef std::pair<regressiontype, classtype> pairtype;

  // Intermediate output type to allow full precision multiplication of jet pt by the ratio
  typedef ap_ufixed<22, 12, AP_TRN, AP_SAT> output_regression_type;
  // Intermediate output type for classification score to be loaded into jet word
  typedef std::array<l1ct::jet_tag_score_t, 8> output_class_type;
  typedef std::pair<regressiontype, output_class_type> outputpairtype;
  void setNNVectorVar();
  outputpairtype EvaluateNNFixed();
  outputpairtype computeFixed(const l1t::PFJet &iJet);

private:
  std::vector<inputtype> NNvectorVar_;
  int fNParticles_;
  unique_ptr<inputtype[]> fPt_;
  unique_ptr<inputtype[]> fPt_rel_;
  unique_ptr<inputtype[]> fDEta_;
  unique_ptr<inputtype[]> fDPhi_;
  unique_ptr<inputtype[]> fPt_log_;
  unique_ptr<inputtype[]> fMass_;
  unique_ptr<inputtype[]> fZ0_;
  unique_ptr<inputtype[]> fDxy_;
  unique_ptr<inputtype[]> fIs_filled_;
  unique_ptr<inputtype[]> fPuppi_weight_;
  unique_ptr<inputtype[]> fEmID_;
  unique_ptr<inputtype[]> fQuality_;

  unique_ptr<inputtype[]> fCharge_;
  unique_ptr<inputtype[]> fId_;
  std::shared_ptr<hls4mlEmulator::Model> modelRef_;

  bool isDebugEnabled_;
};
#endif
