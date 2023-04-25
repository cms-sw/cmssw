#include "L1Trigger/Phase2L1ParticleFlow/interface/l1-converters/hgcalinput_ref.h"

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"
l1ct::HgcalClusterDecoderEmulator::HgcalClusterDecoderEmulator(const edm::ParameterSet &pset)
    : slim_(pset.getParameter<bool>("slim")) {}

#endif

l1ct::HgcalClusterDecoderEmulator::~HgcalClusterDecoderEmulator() {}

l1ct::HadCaloObjEmu l1ct::HgcalClusterDecoderEmulator::decode(const ap_uint<256> &in) const {
  ap_uint<14> w_pt = in(13, 0);
  ap_uint<14> w_empt = in(27, 14);
  ap_int<9> w_eta = in(72, 64);
  ap_int<9> w_phi = in(81, 73);
  ap_uint<10> w_qual = in(115, 106);
  ap_uint<13> w_srrtot = in(213, 201);
  ap_uint<12> w_meanz = in(94, 83);
  // FIXME: we use a spare space in the word for hoe which is not in the current interface
  ap_uint<12> w_hoe = in(127, 116);

  l1ct::HadCaloObjEmu out;
  out.clear();
  out.hwPt = w_pt * l1ct::pt_t(l1ct::Scales::INTPT_LSB);
  out.hwEta = w_eta;
  out.hwPhi = w_phi;  // relative to the region center, at calo
  out.hwEmPt = w_empt * l1ct::pt_t(l1ct::Scales::INTPT_LSB);
  out.hwEmID = w_qual;
  if (!slim_) {
    out.hwSrrTot = w_srrtot * l1ct::srrtot_t(l1ct::Scales::SRRTOT_LSB);
    out.hwMeanZ =
        (w_meanz == 0) ? l1ct::meanz_t(0) : l1ct::meanz_t(w_meanz - l1ct::meanz_t(l1ct::Scales::MEANZ_OFFSET));
    out.hwHoe = w_hoe * l1ct::hoe_t(l1ct::Scales::HOE_LSB);
  }
  return out;
}
