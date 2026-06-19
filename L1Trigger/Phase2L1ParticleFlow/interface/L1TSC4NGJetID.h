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

  typedef ap_fixed<64, 32, AP_RND, AP_SAT, 0> inputtype;

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

  static const int N_candidates = 16;
  static const int N_candidate_features = 21;
  static const int N_candidate_inputs = N_candidates * N_candidate_features;
  static const int N_jet_inputs = 2;
  static const int N_class_outputs = 8;
  static const int N_regression_outputs = 1;

  // Intermediate output type to allow full precision multiplication of jet pt by the ratio
  typedef ap_ufixed<22, 12, AP_TRN, AP_SAT> output_regression_type;
  // Intermediate output type for classification score to be loaded into jet word
  typedef std::array<l1ct::jet_tag_score_t, N_class_outputs> output_class_type;
  typedef std::pair<std::array<output_regression_type, N_regression_outputs>, output_class_type> outputpairtype;

  void setVectors();
  outputpairtype EvaluateNNFixed();
  outputpairtype computeFixed(const l1t::PFJet &iJet);

private:
  std::vector<L1TSC4NGJet::inputtype> candidate_vector_;
  std::vector<L1TSC4NGJet::inputtype> jet_vector_;

  int fNParticles_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fPt_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fPt_rel_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fDEta_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fDPhi_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fPt_log_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fMass_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fZ0_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fDxy_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fIs_filled_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fPuppi_weight_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fEmID_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fQuality_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fEta_;

  std::unique_ptr<L1TSC4NGJet::inputtype[]> fCharge_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fId_;

  L1TSC4NGJet::inputtype fJetPt_;
  L1TSC4NGJet::inputtype fJetEta_;

  std::shared_ptr<hls4mlEmulator::Model> modelRef_;

  bool isDebugEnabled_;
};
#endif
