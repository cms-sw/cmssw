#ifndef L1Trigger_Phase2L1ParticleFlow_newfirmware_gcteminput_ref_h
#define L1Trigger_Phase2L1ParticleFlow_newfirmware_gcteminput_ref_h

#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"

#include "L1Trigger/Phase2L1ParticleFlow/interface/corrector.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/ParametricResolution.h"

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;
}  // namespace edm

namespace l1ct {
  class GctEmClusterDecoderEmulator {
  public:
    GctEmClusterDecoderEmulator() {};

    // only works in CMSSW
    GctEmClusterDecoderEmulator(const edm::ParameterSet &pset);

    GctEmClusterDecoderEmulator(const std::string &corrFile,
                                l1tpf::ParametricResolution::Kind kind,
                                std::vector<float> etas,
                                std::vector<float> offsets,
                                std::vector<float> scales,
                                std::vector<float> ptMins,
                                std::vector<float> ptMaxs)
        : corrector_(corrFile), resol_(kind, etas, offsets, scales, ptMins, ptMaxs) {}

    ~GctEmClusterDecoderEmulator() = default;

    static edm::ParameterSetDescription getParameterSetDescription();

    l1ct::EmCaloObjEmu decode(const l1ct::PFRegionEmu &sector, const ap_uint<64> &in) const;

  private:
    // tools for GCT clusters
    l1tpf::corrector corrector_;
    l1tpf::ParametricResolution resol_;
  };

}  // namespace l1ct

#endif
