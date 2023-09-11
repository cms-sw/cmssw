#include "L1Trigger/Phase2L1ParticleFlow/interface/l1-converters/muonGmtToL1ct_ref.h"

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

l1ct::GMTMuonDecoderEmulator::GMTMuonDecoderEmulator(const edm::ParameterSet &iConfig)
    : z0Scale_(iConfig.getParameter<double>("z0Scale")), dxyScale_(iConfig.getParameter<double>("dxyScale")) {}

edm::ParameterSetDescription l1ct::GMTMuonDecoderEmulator::getParameterSetDescription() {
  edm::ParameterSetDescription description;
  description.add<double>("z0Scale", 1.875);
  description.add<double>("dxyScale", 3.85);
  return description;
}

#endif

l1ct::GMTMuonDecoderEmulator::GMTMuonDecoderEmulator(float z0Scale, float dxyScale)
    : z0Scale_(z0Scale), dxyScale_(dxyScale) {}

l1ct::GMTMuonDecoderEmulator::~GMTMuonDecoderEmulator() {}

l1ct::MuObjEmu l1ct::GMTMuonDecoderEmulator::decode(const ap_uint<64> &in) const {
  typedef ap_ufixed<13, 8, AP_TRN, AP_SAT> gmt_pt_t;

  const int etaPhi_common_bits = 4, etaPhi_extra_bits = 12 - etaPhi_common_bits;
  const int etaPhi_scale = l1ct::Scales::INTPHI_PI >> etaPhi_common_bits;
  const int etaPhi_offs = 1 << (etaPhi_extra_bits - 1);

  const int z0_scale = std::round(z0Scale_ / l1ct::Scales::Z0_LSB);
  const int dxy_scale = std::round(dxyScale_ / l1ct::Scales::DXY_LSB);

  bool gmt_valid = in[0], gmt_chg = in[56];
  ap_uint<13> gmt_ipt = in(16, 1);
  ap_int<13> gmt_phi = in(29, 17);
  ap_int<14> gmt_eta = in(43, 30);
  ap_int<5> gmt_z0 = in(48, 44);
  ap_int<7> gmt_d0 = in(55, 49);
  ap_uint<4> gmt_qual = in(60, 57);

  gmt_pt_t gmt_pt;
  gmt_pt(gmt_pt_t::width - 1, 0) = gmt_ipt(gmt_pt_t::width - 1, 0);  // copy the bits

  l1ct::MuObjEmu out;
  out.clear();
  if (gmt_valid && gmt_pt != 0) {
    // add a shift in order to get the proper rounding
    out.hwPt = gmt_pt + gmt_pt_t(l1ct::Scales::INTPT_LSB / 2);

    out.hwEta = (gmt_eta * etaPhi_scale + etaPhi_offs) >> etaPhi_extra_bits;
    out.hwPhi = (gmt_phi * etaPhi_scale + etaPhi_offs) >> etaPhi_extra_bits;
    out.hwDEta = 0;
    out.hwDPhi = 0;

    out.hwCharge = !gmt_chg;

    out.hwZ0 = gmt_z0 * z0_scale;
    out.hwDxy = gmt_d0 * dxy_scale;

    out.hwQuality = gmt_qual(3, 1);  // drop lowest bit
  }

  return out;
}
