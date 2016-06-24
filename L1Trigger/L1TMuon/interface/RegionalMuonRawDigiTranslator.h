#ifndef RegionalMuonRawDigiTranslator_h
#define RegionalMuonRawDigiTranslator_h

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

namespace l1t {
  class RegionalMuonRawDigiTranslator {
    public:
      static void fillRegionalMuonCand(RegionalMuonCand&, uint32_t, uint32_t, int, tftype);
      static void fillRegionalMuonCand(RegionalMuonCand&, uint64_t, int, tftype);
      static void generatePackedDataWords(const RegionalMuonCand&, uint32_t&, uint32_t&);
      static uint64_t generate64bitDataWord(const RegionalMuonCand&);

      static const unsigned ptMask_ = 0x1FF;
      static const unsigned ptShift_ = 0;
      static const unsigned qualMask_ = 0xF;
      static const unsigned qualShift_ = 9;
      static const unsigned absEtaMask_ = 0xFF;
      static const unsigned absEtaShift_ = 13;
      static const unsigned etaSignShift_ = 21;
      static const unsigned hfMask_ = 0x1;
      static const unsigned hfShift_ = 22;
      static const unsigned absPhiMask_ = 0x7F;
      static const unsigned absPhiShift_ = 23;
      static const unsigned phiSignShift_ = 30;
      static const unsigned signShift_ = 0;
      static const unsigned signValidShift_ = 1;
      static const unsigned trackAddressMask_ = 0x1FFFFFFF;
      static const unsigned trackAddressShift_ = 2;
      // relative shifts within track address
      static const unsigned bmtfTrAddrSegSelMask_ = 0xF;
      static const unsigned bmtfTrAddrSegSelShift_ = 21;
      static const unsigned bmtfTrAddrDetSideShift_ = 20;
      static const unsigned bmtfTrAddrWheelMask_ = 0x3;
      static const unsigned bmtfTrAddrWheelShift_ = 18;
      static const unsigned bmtfTrAddrStat1Mask_ = 0x3;
      static const unsigned bmtfTrAddrStat1Shift_ = 14;
      static const unsigned bmtfTrAddrStat2Mask_ = 0xF;
      static const unsigned bmtfTrAddrStat2Shift_ = 10;
      static const unsigned bmtfTrAddrStat3Mask_ = 0xF;
      static const unsigned bmtfTrAddrStat3Shift_ = 6;
      static const unsigned bmtfTrAddrStat4Mask_ = 0xF;
      static const unsigned bmtfTrAddrStat4Shift_ = 2;

      static const unsigned emtfTrAddrMe1SegShift_ =   0;
      static const unsigned emtfTrAddrMe1ChShift_  =   1;
      static const unsigned emtfTrAddrMe1ChMask_   = 0x7;
      static const unsigned emtfTrAddrMe2SegShift_ =   4;
      static const unsigned emtfTrAddrMe2ChShift_  =   5;
      static const unsigned emtfTrAddrMe2ChMask_   = 0x7;
      static const unsigned emtfTrAddrMe3SegShift_ =   8;
      static const unsigned emtfTrAddrMe3ChShift_  =   9;
      static const unsigned emtfTrAddrMe3ChMask_   = 0x7;
      static const unsigned emtfTrAddrMe4SegShift_ =  12;
      static const unsigned emtfTrAddrMe4ChShift_  =  13;
      static const unsigned emtfTrAddrMe4ChMask_   = 0x7;
      static const unsigned emtfTrAddrTrkNumShift_ =  16;
      static const unsigned emtfTrAddrTrkNumMask_  = 0x3;
      static const unsigned emtfTrAddrBxShift_     =  18;
      static const unsigned emtfTrAddrBxMask_    = 0x7FF;
  };
}

#endif
