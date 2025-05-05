#include "L1Trigger/Phase2L1ParticleFlow/interface/l1-converters/gcteminput_ref.h"

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

// TODO: Currently this only works in CMSSW
l1ct::GctEmClusterDecoderEmulator::GctEmClusterDecoderEmulator(const edm::ParameterSet &iConfig)
    : corrector_(iConfig.getParameter<std::string>("gctEmCorrector"), -1),
      resol_(iConfig.getParameter<edm::ParameterSet>("gctEmResol")) {}

edm::ParameterSetDescription l1ct::GctEmClusterDecoderEmulator::getParameterSetDescription() {
  edm::ParameterSetDescription description;
  description.add<std::string>("gctEmCorrector");
  edm::ParameterSetDescription gctEmResolPSD;
  gctEmResolPSD.add<std::vector<double>>("etaBins");
  gctEmResolPSD.add<std::vector<double>>("offset");
  gctEmResolPSD.add<std::vector<double>>("scale");
  gctEmResolPSD.add<std::string>("kind");
  description.add<edm::ParameterSetDescription>("gctEmResol", gctEmResolPSD);
  return description;
}
#endif

l1ct::GctEmClusterDecoderEmulator::~GctEmClusterDecoderEmulator() {}

l1ct::EmCaloObjEmu l1ct::GctEmClusterDecoderEmulator::decode(const l1ct::PFRegionEmu &sector,
                                                             const ap_uint<64> &in) const {
  constexpr float ETA_RANGE_ONE_SIDE = 1.4841;  // barrel goes from (-1.4841, +1.4841)
  constexpr float ETA_LSB = 2 * ETA_RANGE_ONE_SIDE / 170.;
  constexpr float PHI_LSB = 2 * M_PI / 360.;

  // need to add emid
  l1ct::EmCaloObjEmu calo;
  calo.clear();
  calo.hwPt = pt(in) * l1ct::pt_t(0.5);  // the LSB for GCT objects
  // We add half a crystal both in eta and phi to avoid a bias
  calo.hwEta = l1ct::Scales::makeGlbEta(eta(in) * ETA_LSB + ETA_LSB / 2.);  // at this point eta is abs(globalEta)
  calo.hwPhi = l1ct::Scales::makePhi(phi(in) * PHI_LSB + (PHI_LSB / 2));    // This is already in the local frame

  if (corrector_.valid()) {
    float newpt =
        corrector_.correctedPt(calo.floatPt(), calo.floatPt(), calo.floatEta());  // NOTE: this is still abs(globalEta)
    calo.hwPt = l1ct::Scales::makePtFromFloat(newpt);
  }

  // Note: at this point still
  calo.hwPtErr =
      l1ct::Scales::makePtFromFloat(resol_(calo.floatPt(), calo.floatEta()));  // NOTE: this is still abs(globalEta)

  // hwQual definition:
  // bit 0: standaloneWP: is_iso && is_ss
  // bit 1: looseL1TkMatchWP: is_looseTkiso && is_looseTkss
  // bit 2: photonWP:
  calo.hwEmID = (passes_iso(in) & passes_ss(in)) | ((passes_looseTkiso(in) & passes_looseTkss(in)) << 1) |
                ((passes_looseTkiso(in) & passes_looseTkss(in)) << 2);

  // convert eta to local
  if (sector.hwEtaCenter < 0) {
    calo.hwEta = -calo.hwEta - sector.hwEtaCenter;
  } else {
    calo.hwEta = calo.hwEta - sector.hwEtaCenter;
  }

  return calo;
}
