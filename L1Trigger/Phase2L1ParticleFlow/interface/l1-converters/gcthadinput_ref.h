#ifndef L1Trigger_Phase2L1ParticleFlow_newfirmware_gcthadinput_ref_h
#define L1Trigger_Phase2L1ParticleFlow_newfirmware_gcthadinput_ref_h

#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"

#include "L1Trigger/Phase2L1ParticleFlow/interface/corrector.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/ParametricResolution.h"

// TODO:  add calibration

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;
}  // namespace edm

namespace l1ct {
  class GctHadClusterDecoderEmulator {
  public:
    GctHadClusterDecoderEmulator() {};
    GctHadClusterDecoderEmulator(const edm::ParameterSet &iConfig);
    GctHadClusterDecoderEmulator(const std::string &corrFile,
                                 l1tpf::ParametricResolution::Kind kind,
                                 std::vector<float> etas,
                                 std::vector<float> offsets,
                                 std::vector<float> scales,
                                 std::vector<float> ptMins,
                                 std::vector<float> ptMaxs)
        : corrector_(corrFile), resol_(kind, etas, offsets, scales, ptMins, ptMaxs) {}

    ~GctHadClusterDecoderEmulator() = default;

    static edm::ParameterSetDescription getParameterSetDescription();

    l1ct::HadCaloObjEmu decode(const l1ct::PFRegionEmu &sector, const ap_uint<64> &in) const;

  private:
    // tools for GCT clusters
    l1tpf::corrector corrector_;
    l1tpf::ParametricResolution resol_;
  };
}  // namespace l1ct

#endif
