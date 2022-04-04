#ifndef L1Trigger_Phase2L1ParticleFlow_newfirmware_hgcalinput_ref_h
#define L1Trigger_Phase2L1ParticleFlow_newfirmware_hgcalinput_ref_h

#ifdef CMSSW_GIT_HASH
#include "../../dataformats/layer1_emulator.h"
#else
#include "../../../dataformats/layer1_emulator.h"
#endif

namespace l1ct {
  class HgcalClusterDecoderEmulator {
  public:
    HgcalClusterDecoderEmulator(){};
    ~HgcalClusterDecoderEmulator();
    l1ct::HadCaloObjEmu decode(const ap_uint<256> &in) const;
  };
}  // namespace l1ct

#endif
