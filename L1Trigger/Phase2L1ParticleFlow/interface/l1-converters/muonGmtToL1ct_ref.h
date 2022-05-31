#ifndef L1Trigger_Phase2L1ParticleFlow_newfirmware_muonGmtToL1ct_ref_h
#define L1Trigger_Phase2L1ParticleFlow_newfirmware_muonGmtToL1ct_ref_h

#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"

namespace edm {
  class ParameterSet;
}

namespace l1ct {
  class GMTMuonDecoderEmulator {
  public:
    GMTMuonDecoderEmulator(float z0Scale, float dxyScale);
    GMTMuonDecoderEmulator(const edm::ParameterSet &iConfig);
    ~GMTMuonDecoderEmulator();
    l1ct::MuObjEmu decode(const ap_uint<64> &in) const;

  protected:
    float z0Scale_, dxyScale_;
  };
}  // namespace l1ct

#endif
