#include "L1Trigger/Phase2L1ParticleFlow/interface/l1-converters/gcthadinput_ref.h"

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

l1ct::GctHadClusterDecoderEmulator::GctHadClusterDecoderEmulator(const edm::ParameterSet &pset) {}

edm::ParameterSetDescription l1ct::GctHadClusterDecoderEmulator::getParameterSetDescription() {
  edm::ParameterSetDescription description;
  return description;
}
#endif

l1ct::GctHadClusterDecoderEmulator::~GctHadClusterDecoderEmulator() {}

double l1ct::GctHadClusterDecoderEmulator::fracPart(const double total, const unsigned int hoe) const {
  return total * std::pow(2.0, hoe) / (std::pow(2.0, hoe) + 1);
}

l1ct::HadCaloObjEmu l1ct::GctHadClusterDecoderEmulator::decode(const l1ct::PFRegionEmu &sector,
                                                               const ap_uint<64> &in) const {
  constexpr float ETA_RANGE_ONE_SIDE = 1.4841;  // barrel goes from (-1.4841, +1.4841)
  constexpr float ETA_LSB = 2 * ETA_RANGE_ONE_SIDE / 170.;
  constexpr float PHI_LSB = 2 * M_PI / 360.;

  l1ct::HadCaloObjEmu calo;
  calo.clear();
  calo.hwPt = pt(in) * l1ct::pt_t(0.5);                                     // the LSB for GCT objects
  calo.hwEta = l1ct::Scales::makeGlbEta(eta(in) * ETA_LSB + ETA_LSB / 2.);  // at this point eta is abs(globalEta)
  calo.hwPhi = l1ct::Scales::makePhi(phi(in) * PHI_LSB + (PHI_LSB / 2));    // This is already in the local frame

  // need to add EmPt when it becomes available

  // need to add emid
  calo.hwEmID = 1;

  // convert eta to local
  if (sector.hwEtaCenter < 0) {
    calo.hwEta = -calo.hwEta - sector.hwEtaCenter;
  } else {
    calo.hwEta = calo.hwEta - sector.hwEtaCenter;
  }

  return calo;
}