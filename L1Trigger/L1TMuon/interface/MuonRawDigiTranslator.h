#ifndef MuonRawDigiTranslator_h
#define MuonRawDigiTranslator_h

#include "DataFormats/L1Trigger/interface/Muon.h"

namespace l1t {
  class MuonRawDigiTranslator {
  public:
    static void fillMuon(Muon&, uint32_t, uint32_t, uint32_t, int, unsigned int, int, bool);
    static void fillMuon(Muon&, uint32_t, uint64_t, int, unsigned int, int, bool);
    static void fillMuonCoordinates2016(Muon& mu, uint32_t raw_data_00_31, uint32_t raw_data_32_63);
    static void fillMuonCoordinatesFrom2017(Muon& mu, uint32_t raw_data_00_31, uint32_t raw_data_32_63);
    static void fillMuonCoordinatesRun3(
        Muon& mu, uint32_t raw_data_spare, uint32_t raw_data_00_31, uint32_t raw_data_32_63, int muInBx);
    static void fillIntermediateMuonCoordinatesRun3(Muon& mu, uint32_t raw_data_00_31, uint32_t raw_data_32_63);
    static void generatePackedDataWords(const Muon&, uint32_t&, uint32_t&);
    static uint64_t generate64bitDataWord(const Muon&);
    static int calcHwEta(const uint32_t&, const unsigned, const unsigned);

    static constexpr unsigned ptMask_ = 0x1FF;
    static constexpr unsigned ptShift_ = 10;
    static constexpr unsigned qualMask_ = 0xF;
    static constexpr unsigned qualShift_ = 19;
    static constexpr unsigned absEtaMask_ = 0xFF;
    static constexpr unsigned absEtaShift_ = 21;
    static constexpr unsigned absEtaAtVtxShift_ = 23;
    static constexpr unsigned etaSignShift_ = 29;
    static constexpr unsigned etaAtVtxSignShift_ = 31;
    static constexpr unsigned phiMask_ = 0x3FF;
    static constexpr unsigned phiShift_ = 11;
    static constexpr unsigned phiAtVtxShift_ = 0;
    static constexpr unsigned chargeShift_ = 2;
    static constexpr unsigned chargeValidShift_ = 3;
    static constexpr unsigned tfMuonIndexMask_ = 0x7F;
    static constexpr unsigned tfMuonIndexShift_ = 4;
    static constexpr unsigned isoMask_ = 0x3;
    static constexpr unsigned isoShift_ = 0;
    static constexpr unsigned dxyMask_ = 0x3;
    static constexpr unsigned dxyShift_ = 30;
    static constexpr unsigned ptUnconstrainedMask_ = 0xFF;
    static constexpr unsigned ptUnconstrainedShift_ = 21;
    static constexpr unsigned ptUnconstrainedIntermedidateShift_ = 0;
    static constexpr unsigned absEtaMu1Shift_ = 12;   // For Run-3
    static constexpr unsigned etaMu1SignShift_ = 20;  // For Run-3
    static constexpr unsigned absEtaMu2Shift_ = 21;   // For Run-3
    static constexpr unsigned etaMu2SignShift_ = 29;  // For Run-3
  };
}  // namespace l1t

#endif
