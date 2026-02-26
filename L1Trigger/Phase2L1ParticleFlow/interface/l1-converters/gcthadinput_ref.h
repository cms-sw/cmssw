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

    ~GctHadClusterDecoderEmulator() = default;

    static edm::ParameterSetDescription getParameterSetDescription();

    l1ct::HadCaloObjEmu decode(const l1ct::PFRegionEmu &sector, const ap_uint<64> &in) const;
  };
}  // namespace l1ct

#endif
