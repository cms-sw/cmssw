#ifndef MuonRawDigiTranslator_h
#define MuonRawDigiTranslator_h

#include "DataFormats/L1Trigger/interface/Muon.h"

namespace l1t {
  class MuonRawDigiTranslator {
  public:
    static void fillMuon(Muon& mu,
                         uint32_t raw_data_spare,
                         uint32_t raw_data_00_31,
                         uint32_t raw_data_32_63,
                         int fed,
                         unsigned int fw,
                         int muInBx);
    static void fillMuon(Muon& mu, uint32_t raw_data_spare, uint64_t dataword, int fed, unsigned int fw, int muInBx);
    static void fillIntermediateMuon(Muon& mu, uint32_t raw_data_00_31, uint32_t raw_data_32_63, unsigned int fw);
    static void generatePackedDataWords(const Muon& mu,
                                        uint32_t& raw_data_spare,
                                        uint32_t& raw_data_00_31,
                                        uint32_t& raw_data_32_63,
                                        int fedId,
                                        int fwId,
                                        int muInBx);
    static void generate64bitDataWord(
        const Muon& mu, uint32_t& raw_data_spare, uint64_t& dataword, int fedId, int fwId, int muInBx);
    static int calcHwEta(const uint32_t& raw, const unsigned absEtaShift, const unsigned etaSignShift);

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
    static constexpr unsigned absEtaMu1Shift_ = 13;   // For Run-3
    static constexpr unsigned etaMu1SignShift_ = 21;  // For Run-3
    static constexpr unsigned absEtaMu2Shift_ = 22;   // For Run-3
    static constexpr unsigned etaMu2SignShift_ = 30;  // For Run-3

  private:
    static void fillMuonStableQuantities(Muon& mu, uint32_t raw_data_00_31, uint32_t raw_data_32_63);
    static void fillMuonCoordinates2016(Muon& mu, uint32_t raw_data_00_31, uint32_t raw_data_32_63);
    static void fillMuonCoordinatesFrom2017(Muon& mu, uint32_t raw_data_00_31, uint32_t raw_data_32_63);
    static void fillMuonQuantitiesRun3(Muon& mu,
                                       uint32_t raw_data_spare,
                                       uint32_t raw_data_00_31,
                                       uint32_t raw_data_32_63,
                                       int muInBx,
                                       bool wasSpecialMWGR = false);
    static void fillIntermediateMuonQuantitiesRun3(Muon& mu, uint32_t raw_data_00_31, uint32_t raw_data_32_63);
    static void generatePackedDataWordsRun3(const Muon& mu,
                                            int abs_eta,
                                            int abs_eta_at_vtx,
                                            uint32_t& raw_data_spare,
                                            uint32_t& raw_data_00_31,
                                            uint32_t& raw_data_32_63,
                                            int muInBx,
                                            bool wasSpecialMWGR = false);
  };
}  // namespace l1t

#endif
