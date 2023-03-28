#ifndef MuonRawDigiTranslator_h
#define MuonRawDigiTranslator_h

#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/MuonShower.h"

#include <array>

namespace l1t {
  class MuonRawDigiTranslator {
  public:
    static void fillMuon(Muon& mu,
                         uint32_t raw_data_spare,
                         uint32_t raw_data_00_31,
                         uint32_t raw_data_32_63,
                         int fed,
                         int fw,
                         int muInBx);
    static void fillMuon(Muon& mu, uint32_t raw_data_spare, uint64_t dataword, int fed, int fw, int muInBx);
    static void fillIntermediateMuon(Muon& mu, uint32_t raw_data_00_31, uint32_t raw_data_32_63, int fw);
    static bool showerFired(uint32_t shower_word, int fedId, int fwId);
    static void generatePackedMuonDataWords(const Muon& mu,
                                            uint32_t& raw_data_spare,
                                            uint32_t& raw_data_00_31,
                                            uint32_t& raw_data_32_63,
                                            int fedId,
                                            int fwId,
                                            int muInBx);
    static void generate64bitDataWord(
        const Muon& mu, uint32_t& raw_data_spare, uint64_t& dataword, int fedId, int fwId, int muInBx);
    static std::array<std::array<uint32_t, 4>, 2> getPackedShowerDataWords(const MuonShower& shower,
                                                                           int fedId,
                                                                           int fwId);
    static int calcHwEta(const uint32_t& raw, unsigned absEtaShift, unsigned etaSignShift);

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
    static constexpr unsigned showerShift_ = 29;      // For Run-3
    static constexpr unsigned absEtaMu1Shift_ = 13;   // For Run-3
    static constexpr unsigned etaMu1SignShift_ = 21;  // For Run-3
    static constexpr unsigned absEtaMu2Shift_ = 22;   // For Run-3
    static constexpr unsigned etaMu2SignShift_ = 30;  // For Run-3
    static constexpr int kUgmtFedId = 1402;
    static constexpr int kUgtFedId = 1404;
    static constexpr int kUgmtFwVersionUntil2016 = 0x4010000;
    static constexpr int kUgtFwVersionUntil2016 = 0x10A6;
    static constexpr int kUgmtFwVersionUntil2017 = 0x6000000;
    static constexpr int kUgtFwVersionUntil2017 = 0x1120;
    static constexpr int kUgmtFwVersionRun3Start = 0x6000001;
    static constexpr int kUgtFwVersionUntilRun3Start = 0x1130;
    static constexpr int kUgmtFwVersionFirstWithShowers = 0x7000000;
    static constexpr int kUgtFwVersionFirstWithShowers = 0x113B;
    static constexpr int kUgmtFwVersionShowersFrom2023 = 0x8000000;
    static constexpr int kUgtFwVersionShowersFrom2023 = 0x1150;

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
    static void generatePackedMuonDataWordsRun3(const Muon& mu,
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
