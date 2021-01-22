#ifndef RegionalMuonRawDigiTranslator_h
#define RegionalMuonRawDigiTranslator_h

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

namespace l1t {
  class RegionalMuonRawDigiTranslator {
  public:
    static void fillRegionalMuonCand(
        RegionalMuonCand& mu, uint32_t raw_data_00_31, uint32_t raw_data_32_63, int proc, tftype tf, bool isKalman);
    static void fillRegionalMuonCand(RegionalMuonCand& mu, uint64_t dataword, int proc, tftype tf, bool isKalman);
    static void generatePackedDataWords(const RegionalMuonCand& mu,
                                        uint32_t& raw_data_00_31,
                                        uint32_t& raw_data_32_63,
                                        bool isKalman);
    static uint64_t generate64bitDataWord(const RegionalMuonCand& mu, bool isKalman);
    static int generateRawTrkAddress(const RegionalMuonCand&, bool isKalman);

    static constexpr unsigned ptMask_ = 0x1FF;
    static constexpr unsigned ptShift_ = 0;
    static constexpr unsigned qualMask_ = 0xF;
    static constexpr unsigned qualShift_ = 9;
    static constexpr unsigned absEtaMask_ = 0xFF;
    static constexpr unsigned absEtaShift_ = 13;
    static constexpr unsigned etaSignShift_ = 21;
    static constexpr unsigned hfMask_ = 0x1;
    static constexpr unsigned hfShift_ = 22;
    static constexpr unsigned absPhiMask_ = 0x7F;
    static constexpr unsigned absPhiShift_ = 23;
    static constexpr unsigned phiSignShift_ = 30;
    static constexpr unsigned signShift_ = 0;
    static constexpr unsigned signValidShift_ = 1;
    static constexpr unsigned dxyMask_ = 0x3;
    static constexpr unsigned dxyShift_ = 2;
    static constexpr unsigned ptUnconstrainedMask_ = 0xFF;
    static constexpr unsigned ptUnconstrainedShift_ = 23;
    static constexpr unsigned trackAddressMask_ = 0x1FFFFFFF;
    static constexpr unsigned trackAddressShift_ = 2;
    // relative shifts within track address
    static constexpr unsigned bmtfTrAddrSegSelMask_ = 0xF;
    static constexpr unsigned bmtfTrAddrSegSelShift_ = 21;
    static constexpr unsigned bmtfTrAddrDetSideShift_ = 20;
    static constexpr unsigned bmtfTrAddrWheelMask_ = 0x3;
    static constexpr unsigned bmtfTrAddrWheelShift_ = 18;
    static constexpr unsigned bmtfTrAddrStat1Mask_ = 0x3;
    static constexpr unsigned bmtfTrAddrStat1Shift_ = 14;
    static constexpr unsigned bmtfTrAddrStat2Mask_ = 0xF;
    static constexpr unsigned bmtfTrAddrStat2Shift_ = 10;
    static constexpr unsigned bmtfTrAddrStat3Mask_ = 0xF;
    static constexpr unsigned bmtfTrAddrStat3Shift_ = 6;
    static constexpr unsigned bmtfTrAddrStat4Mask_ = 0xF;
    static constexpr unsigned bmtfTrAddrStat4Shift_ = 2;

    static constexpr unsigned emtfTrAddrMe1SegShift_ = 0;
    static constexpr unsigned emtfTrAddrMe1ChShift_ = 1;
    static constexpr unsigned emtfTrAddrMe1ChMask_ = 0x7;
    static constexpr unsigned emtfTrAddrMe2SegShift_ = 4;
    static constexpr unsigned emtfTrAddrMe2ChShift_ = 5;
    static constexpr unsigned emtfTrAddrMe2ChMask_ = 0x7;
    static constexpr unsigned emtfTrAddrMe3SegShift_ = 8;
    static constexpr unsigned emtfTrAddrMe3ChShift_ = 9;
    static constexpr unsigned emtfTrAddrMe3ChMask_ = 0x7;
    static constexpr unsigned emtfTrAddrMe4SegShift_ = 12;
    static constexpr unsigned emtfTrAddrMe4ChShift_ = 13;
    static constexpr unsigned emtfTrAddrMe4ChMask_ = 0x7;
    static constexpr unsigned emtfTrAddrTrkNumShift_ = 16;
    static constexpr unsigned emtfTrAddrTrkNumMask_ = 0x3;
    static constexpr unsigned emtfTrAddrBxShift_ = 18;
    static constexpr unsigned emtfTrAddrBxMask_ = 0x7FF;

    static constexpr unsigned omtfTrAddrLayersShift_ = 0;
    static constexpr unsigned omtfTrAddrLayersMask_ = 0x3FFFF;
    static constexpr unsigned omtfTrAddrWeightShift_ = 18;
    static constexpr unsigned omtfTrAddrWeightMask_ = 0x1F;
  };
}  // namespace l1t

#endif
