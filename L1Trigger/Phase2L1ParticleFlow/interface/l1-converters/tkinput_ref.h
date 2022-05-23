#ifndef L1Trigger_Phase2L1ParticleFlow_l1converters_tracks_tkinput_ref_h
#define L1Trigger_Phase2L1ParticleFlow_l1converters_tracks_tkinput_ref_h

#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"
#include <cstdio>
#include <algorithm>

namespace edm {
  class ParameterSet;
}

namespace l1ct {
  class TrackInputEmulator {
  public:
    enum class Region { Barrel, Endcap, Any };  // but only Endcap supported for now

    /// encoding used in the digitized track word
    enum class Encoding {
      Stepping,  // J = sign(F) * floor(abs(F)/LSB);  F = sign(J) * ( abs(J) + 0.5 ) * LSB
      Biased,    // J = floor(F/LSB);  F = (J + 0.5)*LSB
      Unbiased   // J = round(F/LSB);  F = J * LSB
    };

    TrackInputEmulator(const edm::ParameterSet &iConfig);
    TrackInputEmulator(Region = Region::Endcap, Encoding encoding = Encoding::Stepping, bool bitwise = true);

    std::pair<l1ct::TkObjEmu, bool> decodeTrack(ap_uint<96> tkword, const l1ct::PFRegionEmu &sector) const {
      return decodeTrack(tkword, sector, bitwise_);
    }
    std::pair<l1ct::TkObjEmu, bool> decodeTrack(ap_uint<96> tkword,
                                                const l1ct::PFRegionEmu &sector,
                                                bool bitwise) const;

    //== Unpackers ==
    static bool valid(const ap_uint<96> &tkword) { return tkword[95]; }
    static bool charge(const ap_uint<96> &tkword) { return !tkword[94]; }

    static ap_int<15> signedRinv(const ap_uint<96> &tkword) { return ap_int<15>(tkword(94, 80)); }
    static ap_int<12> signedZ0(const ap_uint<96> &tkword) { return ap_int<12>(tkword(47, 36)); }
    static ap_int<16> signedTanl(const ap_uint<96> &tkword) { return ap_int<16>(tkword(63, 48)); }
    static ap_int<12> signedPhi(const ap_uint<96> &tkword) { return ap_int<12>(tkword(79, 68)); }

    //=== Floating point conversions ===
    /// just unpack tanl to a float
    float floatTanl(ap_int<16> tanl) const { return toFloat_(tanl) / (1 << 12); }

    /// convert track-word int tanl into float eta (at vertex) in radiants (exact)
    float floatEta(ap_int<16> tanl) const;

    /// convert track-word Rinv into float pt (almost exact)
    float floatPt(ap_int<15> Rinv) const;

    /// convert track-word int phi into float phi (at vertex) in radiants (exact)
    float floatPhi(ap_int<12> phi) const;

    /// convert track-word int z0 into float z0 in cm (exact)
    float floatZ0(ap_int<12> z0) const;

    //=== Configuration of floating point conversions
    void setRinvToPtFactor(float rInvToPt) { rInvToPt_ = rInvToPt; }
    void setPhiScale(float phiScale) { phiScale_ = phiScale; }
    void setZ0Scale(float z0Scale) { z0Scale_ = z0Scale; }

    //=== Bitwise accurate conversions ===
    l1ct::pt_t convPt(ap_int<15> Rinv) const;

    /// convert track-word int tanl into eta *at vertex* in layer 1 units
    l1ct::glbeta_t convEta(ap_int<16> tanl) const;

    /// convert track-word int phi into phi *at vertex* in layer 1 units
    l1ct::phi_t convPhi(ap_int<12> phi) const;

    l1ct::z0_t convZ0(ap_int<12> z0) const;

    //=== Configuration for bitwise accurate conversions ===
    void configPt(int lutBits);

    void configEta(int lutBits, int preOffs, int shift, int postOffs, bool lutSigned, bool endcap);

    void configPhi(int bits);

    void configZ0(int bits);

    //=== Track propagation to calo (float parametrization, no rounding) ===
    //
    // barrel DEta propagation, in layer-1 units (float parameterization, no rounding)
    float floatDEtaBarrel(ap_int<12> z0, ap_int<15> Rinv, ap_int<16> tanl) const;
    // barrel DPhi propagation, in layer-1 units (float parameterization, no rounding)
    float floatDPhiBarrel(ap_int<12> z0, ap_int<15> Rinv, ap_int<16> tanl) const;

    void setDEtaBarrelParams(float pZ0) { dEtaBarrelParamZ0_ = pZ0; }
    void setDPhiBarrelParams(float pC) { dPhiBarrelParamC_ = pC; }

    //=== Track propagation to calo (bitwise accurate) ===
    l1ct::tkdeta_t calcDEtaBarrel(ap_int<12> z0, ap_int<15> Rinv, ap_int<16> tanl) const;
    l1ct::tkdphi_t calcDPhiBarrel(ap_int<12> z0, ap_int<15> Rinv, ap_int<16> tanl) const;

    //=== Configuration of bitwise accurate propagation to calo ===
    void configDEtaBarrel(int dEtaBarrelBits, int dEtaBarrelZ0PreShift, int dEtaBarrelZ0PostShift, float offs = 0);
    void configDPhiBarrel(int dPhiBarrelBits, int dPhiBarrelRInvPreShift, int dPhiBarrelRInvPostShift, float offs = 0);

    // endcap DEta propagation, in layer-1 units (float parameterization, no rounding)
    float floatDEtaHGCal(ap_int<12> z0, ap_int<15> Rinv, ap_int<16> tanl) const;
    // endcap DPhi propagation, in layer-1 units (float parameterization, no rounding)
    float floatDPhiHGCal(ap_int<12> z0, ap_int<15> Rinv, ap_int<16> tanl) const;

    void setDEtaHGCalParams(float pZ0, float pRinv2C, float pRinv2ITanl1, float pRinv2ITanl2) {
      dEtaHGCalParamZ0_ = pZ0;
      dEtaHGCalParamRInv2C_ = pRinv2C;
      dEtaHGCalParamRInv2ITanl1_ = pRinv2ITanl1;
      dEtaHGCalParamRInv2ITanl2_ = pRinv2ITanl2;
    }
    void setDPhiHGCalParams(float pZ0, float pC) {
      dPhiHGCalParamZ0_ = pZ0;
      dPhiHGCalParamC_ = pC;
    }

    //=== Track propagation to calo (bitwise accurate) ===
    l1ct::tkdeta_t calcDEtaHGCal(ap_int<12> z0, ap_int<15> Rinv, ap_int<16> tanl) const;
    l1ct::tkdphi_t calcDPhiHGCal(ap_int<12> z0, ap_int<15> Rinv, ap_int<16> tanl) const;

    //=== Configuration of bitwise accurate propagation to calo ===
    void configDEtaHGCal(int dEtaHGCalBits,
                         int dEtaHGCalZ0PreShift,
                         int dEtaHGCalRInvPreShift,
                         int dEtaHGCalLUTBits,
                         int dEtaHGCalLUTShift,
                         float offs = 0);
    void configDPhiHGCal(int dPhiHGCalBits,
                         int dPhiHGCalZ0PreShift,
                         int dPhiHGCalZ0PostShift,
                         int dPhiHGCalRInvShift,
                         int dPhiHGCalTanlInvShift,
                         int dPhiHGCalTanlLUTBits,
                         float offs = 0);

    /// conservative cut to select tracks that may have |eta| > 1.25 or |calo eta| > 1.25
    static bool mayReachHGCal(ap_int<16> tanl) { return (tanl > 6000) || (tanl < -6000); }
    /// conservative cut to avoid filling LUTs outside of the tracker range
    static bool withinTracker(ap_int<16> tanl) { return (-25000 < tanl) && (tanl < 25000); }
    /// conservative cut to avoid filling LUTs outside of the barrel range
    static bool withinBarrel(ap_int<16> tanl) { return (-13000 < tanl) && (tanl < 13000); }

    void setDebug(bool debug = true) { debug_ = debug; }

    // access to bare LUTs
    //const std::vector<int> &dEtaBarrelLUT() const { return dEtaBarrelLUT_; }
    //const std::vector<int> &dPhiBarrelTanlLUT() const { return dPhiBarrelTanlLUT_; }
    const std::vector<int> &dEtaHGCalLUT() const { return dEtaHGCalLUT_; }
    const std::vector<int> &dPhiHGCalTanlLUT() const { return dPhiHGCalTanlLUT_; }
    const std::vector<int> &tanlLUT() const { return tanlLUT_; }
    const std::vector<l1ct::pt_t> &ptLUT() const { return ptLUT_; }

  protected:
    // utilities
    template <int N>
    inline float toFloat_(ap_int<N> signedVal) const {
      float ret = signedVal.to_float();
      switch (encoding_) {
        case Encoding::Stepping:
          return (signedVal >= 0 ? ret + 0.5 : ret - 0.5);
        case Encoding::Biased:
          return ret + 0.5;
        default:
          return ret;
      }
    }

    /// Region for which the emulation is configured
    Region region_;

    /// Encoding used for track word inputs
    Encoding encoding_;

    /// Whether to run the bitwise accurate or floating point conversions
    bool bitwise_;

    /// Main constants
    float rInvToPt_, phiScale_, z0Scale_;

    /// Parameters for track propagation in floating point
    float dEtaBarrelParamZ0_;
    float dPhiBarrelParamC_;

    /// Parameters for track propagation in floating point
    float dEtaHGCalParamZ0_, dEtaHGCalParamRInv2C_, dEtaHGCalParamRInv2ITanl1_, dEtaHGCalParamRInv2ITanl2_;
    float dPhiHGCalParamZ0_, dPhiHGCalParamC_;

    // vtx phi conversion parameters
    int vtxPhiMult_, vtxPhiOffsPos_, vtxPhiOffsNeg_, vtxPhiBitShift_;

    // z0 conversion parameters
    int z0Mult_, z0OffsPos_, z0OffsNeg_, z0BitShift_;

    // deta parameters in barrel region
    int dEtaBarrelBits_, dEtaBarrelZ0PreShift_, dEtaBarrelZ0PostShift_, dEtaBarrelOffs_, dEtaBarrelZ0_;

    // dphi parameters in barrel region
    int dPhiBarrelBits_, dPhiBarrelRInvPreShift_, dPhiBarrelRInvPostShift_, dPhiBarrelOffs_, dPhiBarrelC_;

    // deta parameters in hgcal region
    int dEtaHGCalBits_, dEtaHGCalZ0PreShift_, dEtaHGCalZ0_, dEtaHGCalRInvPreShift_, dEtaHGCalTanlShift_,
        dEtaHGCalLUTShift_, dEtaHGCalTanlTermOffs_, dEtaHGCalTanlTermShift_, dEtaHGCalOffs_;
    std::vector<int> dEtaHGCalLUT_;

    // dphi parameters in hgcal region
    int dPhiHGCalBits_, dPhiHGCalZ0PreShift_, dPhiHGCalZ0_, dPhiHGCalZ0PostShift_, dPhiHGCalRInvShift_,
        dPhiHGCalTanlShift_, dPhiHGCalTanlInvShift_, dPhiHGCalPreOffs_, dPhiHGCalOffs_;
    std::vector<int> dPhiHGCalTanlLUT_;

    // tanl to eta LUT parameters
    int tanlLUTPreOffs_, tanlLUTShift_, tanlLUTPostOffs_;
    std::vector<int> tanlLUT_;
    bool tanlLUTSigned_;

    // Rinv to pR LUT parameters
    int ptLUTShift_;
    std::vector<l1ct::pt_t> ptLUT_;

    /// enable debug printout in some metods
    bool debug_;
  };
}  // namespace l1ct

#endif
