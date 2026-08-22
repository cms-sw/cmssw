#include "L1Trigger/Phase2L1ParticleFlow/interface/l1-converters/gcthadinput_ref.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/GCTHadDigiCluster.h"

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

l1ct::GctHadClusterDecoderEmulator::GctHadClusterDecoderEmulator(const edm::ParameterSet &iConfig)
    : corrector_(iConfig.getParameter<std::string>("gctHadCorrector"), -1),
      resol_(iConfig.getParameter<edm::ParameterSet>("gctHadResol")) {}

edm::ParameterSetDescription l1ct::GctHadClusterDecoderEmulator::getParameterSetDescription() {
  edm::ParameterSetDescription description;
  description.add<std::string>("gctHadCorrector");
  edm::ParameterSetDescription gctHadResolPSD;
  gctHadResolPSD.add<std::vector<double>>("etaBins");
  gctHadResolPSD.add<std::vector<double>>("offset");
  gctHadResolPSD.add<std::vector<double>>("scale");
  gctHadResolPSD.add<std::string>("kind");
  description.add<edm::ParameterSetDescription>("gctHadResol", gctHadResolPSD);
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

  calo.hwEmPt = inclus.ecal() * inclus.ptLSB();

  if (corrector_.valid()) {
    float newpt = corrector_.correctedPt(
        calo.floatPt(), calo.floatEmPt(), calo.floatEta());  // NOTE: this is still abs(globalEta)
    calo.hwPt = l1ct::Scales::makePtFromFloat(newpt);
  }

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
