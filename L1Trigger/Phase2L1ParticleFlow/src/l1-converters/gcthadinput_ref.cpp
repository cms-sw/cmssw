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
  l1ct::HadCaloObjEmu calo;
  calo.clear();
  calo.hwPt = pt(in) * l1ct::pt_t(0.5);  // the LSB for GCT objects
  calo.hwEta = eta(in) * 4;              // at this point eta is abs(globalEta)
  calo.hwPhi = phi(in) * 4;

  // The proposal is that hoe is going away, to be replaced by EmPt, which is not there yet.

  // // TODO:  this should change
  // // need to add empt
  // ap_uint<4> hoeVal = hoe(in);
  // // the lsb indicates what's bigger, EM or HAD
  // auto isEMBigger = static_cast<bool>(hoeVal[0]);
  // // This is not quite true. If HAD energy goes down to 0, then it flips and says that HAD is bigger
  // ap_uint<3> hoe = hoeVal(3, 1);

  // if (isEMBigger) {
  //   auto em = fracPart(calo.hwPt.to_double(), hoe.to_uint());
  //   calo.hwEmPt = em;
  // } else {
  //   pt_t had = fracPart(calo.hwPt.to_double(), hoe.to_uint());
  //   calo.hwEmPt = calo.hwPt - had;
  // }

  // calo.hwHoe = hoe.to_uint();  // might need to scale

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