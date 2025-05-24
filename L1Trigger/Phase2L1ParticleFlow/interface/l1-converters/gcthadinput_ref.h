#ifndef L1Trigger_Phase2L1ParticleFlow_newfirmware_gcthadinput_ref_h
#define L1Trigger_Phase2L1ParticleFlow_newfirmware_gcthadinput_ref_h

#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"

// TODO:  add calibration

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;
}  // namespace edm

namespace l1ct {
  class GctHadClusterDecoderEmulator {
  public:
    GctHadClusterDecoderEmulator() {};
    GctHadClusterDecoderEmulator(const edm::ParameterSet &pset);

    ~GctHadClusterDecoderEmulator();

    static edm::ParameterSetDescription getParameterSetDescription();

    l1ct::HadCaloObjEmu decode(const l1ct::PFRegionEmu &sector, const ap_uint<64> &in) const;

  private:
    double fracPart(const double total, const unsigned int hoe) const;
    ap_uint<12> pt(const ap_uint<64> &in) const { return in.range(11, 0); }

    // crystal eta (unsigned 7 bits)
    ap_uint<7> eta(const ap_uint<64> &in) const { return (ap_uint<7>)in.range(18, 12); }

    // crystal phi (signed 7 bits)
    ap_int<7> phi(const ap_uint<64> &in) const { return (ap_int<7>)in.range(25, 19); }

    // HoE value
    ap_uint<4> hoe(const ap_uint<64> &in) const { return in.range(30, 26); }
  };
}  // namespace l1ct

#endif
