#ifndef L1Trigger_Phase2L1ParticleFlow_newfirmware_gcteminput_ref_h
#define L1Trigger_Phase2L1ParticleFlow_newfirmware_gcteminput_ref_h

#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/GCTEmDigiCluster.h"

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

    ~GctEmClusterDecoderEmulator();

    static edm::ParameterSetDescription getParameterSetDescription();

    l1ct::EmCaloObjEmu decode(const l1ct::DetectorSector<l1ct::EmCaloObjEmu> &sec, const ap_uint<64> &in) const;

  private:
    ap_uint<12> pt(const ap_uint<64> &in) const { return in.range(11, 0); }

    // crystal eta (unsigned 7 bits)
    ap_uint<7> eta(const ap_uint<64> &in) const { return (ap_uint<7>)in.range(18, 12); }

    // crystal phi (signed 7 bits)
    ap_int<7> phi(const ap_uint<64> &in) const { return (ap_int<7>)in.range(25, 19); }

    // iso flag: two bits, least significant bit is the standalone WP (true or false), second bit is the looseTk WP (true or false)
    // e.g. 0b01 : standalone iso flag passed, loose Tk iso flag did not pass
    ap_uint<2> isoFlags(const ap_uint<64> &in) const { return in.range(36, 35); }
    bool passes_iso(const ap_uint<64> &in) const { return (isoFlags(in) & 0x1); }         // standalone iso WP
    bool passes_looseTkiso(const ap_uint<64> &in) const { return (isoFlags(in) & 0x2); }  // loose Tk iso WP

    // shower shape shape flag: two bits, least significant bit is the standalone WP, second bit is the looseTk WP
    // e.g. 0b01 : standalone shower shape flag passed, loose Tk shower shape flag did not pass
    ap_uint<2> shapeFlags(const ap_uint<64> &in) const { return in.range(49, 48); }

    bool passes_ss(const ap_uint<64> &in) const { return (shapeFlags(in) & 0x1); }         // standalone shower shape WP
    bool passes_looseTkss(const ap_uint<64> &in) const { return (shapeFlags(in) & 0x2); }  // loose Tk shower shape WP

    // tools for GCT clusters
    l1tpf::corrector corrector_;
    l1tpf::ParametricResolution resol_;
  };

}  // namespace l1ct

#endif
