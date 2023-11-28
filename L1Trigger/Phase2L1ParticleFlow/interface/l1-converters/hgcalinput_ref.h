#ifndef L1Trigger_Phase2L1ParticleFlow_newfirmware_hgcalinput_ref_h
#define L1Trigger_Phase2L1ParticleFlow_newfirmware_hgcalinput_ref_h

#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;
}  // namespace edm

namespace l1ct {
  class HgcalClusterDecoderEmulator {
    bool slim_;

  public:
    HgcalClusterDecoderEmulator(bool slim = false) : slim_{slim} {};
    HgcalClusterDecoderEmulator(const edm::ParameterSet &pset);

    ~HgcalClusterDecoderEmulator();

    static edm::ParameterSetDescription getParameterSetDescription();

    l1ct::HadCaloObjEmu decode(const ap_uint<256> &in) const;
  };
}  // namespace l1ct

#endif
