#include "L1Trigger/Phase2L1ParticleFlow/interface/l1-converters/gcthadinput_ref.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/GCTHadDigiCluster.h"

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

l1ct::GctHadClusterDecoderEmulator::GctHadClusterDecoderEmulator(const edm::ParameterSet &pset) {}

edm::ParameterSetDescription l1ct::GctHadClusterDecoderEmulator::getParameterSetDescription() {
  edm::ParameterSetDescription description;
  return description;
}
#endif

l1ct::HadCaloObjEmu l1ct::GctHadClusterDecoderEmulator::decode(const l1ct::PFRegionEmu &sector,
                                                               const ap_uint<64> &in) const {
  constexpr float ETA_RANGE_ONE_SIDE = 1.4841;  // barrel goes from (-1.4841, +1.4841)
  constexpr float ETA_LSB = 2 * ETA_RANGE_ONE_SIDE / 170.;
  constexpr float PHI_LSB = 2 * M_PI / 360.;

  l1tp2::GCTHadDigiCluster inclus(in);

  l1ct::HadCaloObjEmu calo;
  calo.clear();
  calo.hwPt = inclus.pt() * inclus.ptLSB();                                      // the LSB for GCT objects
  calo.hwEta = l1ct::Scales::makeGlbEta(inclus.eta() * ETA_LSB + ETA_LSB / 2.);  // at this point eta is abs(globalEta)
  calo.hwPhi = l1ct::Scales::makePhi(inclus.phi() * PHI_LSB + (PHI_LSB / 2));    // This is already in the local frame

  // need to add EmPt when it becomes available
  calo.hwEmPt = inclus.ecal() * inclus.ptLSB();

  // need to add emid, default to zero for had/PF clusters from GCT
  calo.hwEmID = inclus.fb();

  // convert eta to local
  if (sector.hwEtaCenter < 0) {
    calo.hwEta = -calo.hwEta - sector.hwEtaCenter;
  } else {
    calo.hwEta = calo.hwEta - sector.hwEtaCenter;
  }

  return calo;
}
