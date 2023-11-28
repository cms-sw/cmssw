#ifndef L1ScoutingRawToDigi_shifts_h
#define L1ScoutingRawToDigi_shifts_h

#include <cstdint>

namespace l1ScoutingRun3 {

  namespace ugmt {
    // struct shifts{
    struct shiftsMuon {
      // bx word: 16 bits used for actual bx, MS 4 bits are muon type
      // 0xf intermediate,
      // 0x0 final
      // following 4 bits for link id
      static constexpr uint32_t bx = 0;
      static constexpr uint32_t interm = 31;  // updated for new run3 format (tj)
      // shifts for muon 64 bits
      static constexpr uint32_t phiext = 0;
      static constexpr uint32_t pt = 10;
      static constexpr uint32_t qual = 19;
      static constexpr uint32_t etaext = 23;
      static constexpr uint32_t iso = 0;
      static constexpr uint32_t chrg = 2;
      static constexpr uint32_t chrgv = 3;
      static constexpr uint32_t index = 4;
      static constexpr uint32_t phi = 11;
      static constexpr uint32_t eta1 = 13;
      static constexpr uint32_t eta2 = 22;
      static constexpr uint32_t ptuncon = 21;
      static constexpr uint32_t dxy = 30;
    };
  }  // namespace ugmt

  namespace demux {
    // struct shiftsCaloJet{
    struct shiftsJet {
      static constexpr uint32_t ET = 0;
      static constexpr uint32_t eta = 11;
      static constexpr uint32_t phi = 19;
      static constexpr uint32_t disp = 27;
      static constexpr uint32_t qual = 28;
    };

    // struct shiftsCaloEGamma{
    struct shiftsEGamma {
      static constexpr uint32_t ET = 0;
      static constexpr uint32_t eta = 9;
      static constexpr uint32_t phi = 17;
      static constexpr uint32_t iso = 25;
    };

    // struct shiftsCaloTau{
    struct shiftsTau {
      static constexpr uint32_t ET = 0;
      static constexpr uint32_t eta = 9;
      static constexpr uint32_t phi = 17;
      static constexpr uint32_t iso = 25;
    };

    // struct shiftsCaloESums{
    struct shiftsESums {
      static constexpr uint32_t ETEt = 0;  // Et of ET object
      static constexpr uint32_t ETEttem = 12;
      static constexpr uint32_t ETMinBiasHF = 28;

      static constexpr uint32_t HTEt = 0;  // Et of HT object
      static constexpr uint32_t HTtowerCount = 12;
      static constexpr uint32_t HTMinBiasHF = 28;

      static constexpr uint32_t ETmissEt = 0;
      static constexpr uint32_t ETmissPhi = 12;
      static constexpr uint32_t ETmissASYMET = 20;
      static constexpr uint32_t ETmissMinBiasHF = 28;

      static constexpr uint32_t HTmissEt = 0;
      static constexpr uint32_t HTmissPhi = 12;
      static constexpr uint32_t HTmissASYMHT = 20;
      static constexpr uint32_t HTmissMinBiasHF = 28;

      static constexpr uint32_t ETHFmissEt = 0;
      static constexpr uint32_t ETHFmissPhi = 12;
      static constexpr uint32_t ETHFmissASYMETHF = 20;
      static constexpr uint32_t ETHFmissCENT = 28;

      static constexpr uint32_t HTHFmissEt = 0;
      static constexpr uint32_t HTHFmissPhi = 12;
      static constexpr uint32_t HTHFmissASYMHTHF = 20;
      static constexpr uint32_t HTHFmissCENT = 28;
    };
  }  // namespace demux

  struct header_shifts {
    static constexpr uint32_t bxmatch = 24;
    static constexpr uint32_t mAcount = 16;
    static constexpr uint32_t orbitmatch = 8;
    static constexpr uint32_t warningTestEnabled = 8;
    static constexpr uint32_t mBcount = 0;
    static constexpr uint32_t sBmtfCount = 0;
  };

}  // namespace l1ScoutingRun3
#endif  // L1ScoutingRawToDigi_shifts_h
