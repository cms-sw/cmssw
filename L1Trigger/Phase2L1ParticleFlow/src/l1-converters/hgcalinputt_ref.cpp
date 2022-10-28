#include "L1Trigger/Phase2L1ParticleFlow/interface/l1-converters/hgcalinput_ref.h"

l1ct::HgcalClusterDecoderEmulator::~HgcalClusterDecoderEmulator() {}

l1ct::HadCaloObjEmu l1ct::HgcalClusterDecoderEmulator::decode(const ap_uint<256> &in) const {
  ap_uint<14> w_pt = in(13, 0);
  ap_uint<14> w_empt = in(27, 14);
  ap_int<9> w_eta = in(72, 64);
  ap_int<9> w_phi = in(81, 73);
  ap_uint<10> w_qual = in(115, 106);

  l1ct::HadCaloObjEmu out;
  out.clear();
  out.hwPt = w_pt * l1ct::pt_t(l1ct::Scales::INTPT_LSB);
  out.hwEta = w_eta;
  out.hwPhi = w_phi;  // relative to the region center, at calo
  out.hwEmPt = w_empt * l1ct::pt_t(l1ct::Scales::INTPT_LSB);
  out.hwEmID = w_qual;

  return out;
}
