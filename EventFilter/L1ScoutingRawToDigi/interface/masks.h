#ifndef L1ScoutingRawToDigi_masks_h
#define L1ScoutingRawToDigi_masks_h

#include <cstdint>
#include "shifts.h"

namespace l1ScoutingRun3 {

  namespace ugmt {
    struct masksMuon {
      // bx word: 16 bits used for actual bx, MS 4 bits are muon type
      // 0xf intermediate,
      // 0x0 final
      // following 4 bits for link id
      static constexpr uint32_t bx = 0x1fff;
      static constexpr uint32_t interm = 0x0001;
      //masks for muon 64 bits
      static constexpr uint32_t phiext = 0x03ff;
      static constexpr uint32_t pt = 0x01ff;
      static constexpr uint32_t ptuncon = 0x00ff;  // 8 bits
      static constexpr uint32_t qual = 0x000f;
      static constexpr uint32_t etaext = 0x01ff;
      static constexpr uint32_t etaextv = 0x00ff;
      static constexpr uint32_t etaexts = 0x0100;
      static constexpr uint32_t iso = 0x0003;
      static constexpr uint32_t chrg = 0x0001;
      static constexpr uint32_t chrgv = 0x0001;
      static constexpr uint32_t index = 0x007f;
      static constexpr uint32_t phi = 0x03ff;
      static constexpr uint32_t eta = 0x01ff;
      static constexpr uint32_t etav = 0x00ff;
      static constexpr uint32_t etas = 0x0100;
      static constexpr uint32_t dxy = 0x0003;
    };
  }  // namespace ugmt

  namespace demux {

    struct masksJet {
      static constexpr uint32_t ET = 0x07ff;
      static constexpr uint32_t eta = 0x00ff;
      static constexpr uint32_t phi = 0x00ff;
      static constexpr uint32_t disp = 0x0001;
      static constexpr uint32_t qual = 0x0003;
    };

    struct masksEGamma {
      static constexpr uint32_t ET = 0x01ff;
      static constexpr uint32_t eta = 0x00ff;
      static constexpr uint32_t phi = 0x00ff;
      static constexpr uint32_t iso = 0x0003;
    };

    struct masksTau {
      static constexpr uint32_t ET = 0x01ff;
      static constexpr uint32_t eta = 0x00ff;
      static constexpr uint32_t phi = 0x00ff;
      static constexpr uint32_t iso = 0x0003;
    };

    struct masksESums {
      static constexpr uint32_t ETEt = 0x0fff;  // Et of ET object
      static constexpr uint32_t ETEttem = 0x0fff;
      static constexpr uint32_t ETMinBiasHF = 0x000f;

      static constexpr uint32_t HTEt = 0x0fff;  // Et of HT object
      static constexpr uint32_t HTtowerCount = 0x1fff;
      static constexpr uint32_t HTMinBiasHF = 0x000f;

      static constexpr uint32_t ETmissEt = 0x0fff;
      static constexpr uint32_t ETmissPhi = 0x00ff;
      static constexpr uint32_t ETmissASYMET = 0x00ff;
      static constexpr uint32_t ETmissMinBiasHF = 0x000f;

      static constexpr uint32_t HTmissEt = 0x0fff;
      static constexpr uint32_t HTmissPhi = 0x00ff;
      static constexpr uint32_t HTmissASYMHT = 0x00ff;
      static constexpr uint32_t HTmissMinBiasHF = 0x000f;

      static constexpr uint32_t ETHFmissEt = 0x0fff;
      static constexpr uint32_t ETHFmissPhi = 0x00ff;
      static constexpr uint32_t ETHFmissASYMETHF = 0x00ff;
      static constexpr uint32_t ETHFmissCENT = 0x0003;

      static constexpr uint32_t HTHFmissEt = 0x0fff;
      static constexpr uint32_t HTHFmissPhi = 0x00ff;
      static constexpr uint32_t HTHFmissASYMHTHF = 0x00ff;
      static constexpr uint32_t HTHFmissCENT = 0x0003;
    };
  }  // namespace demux

  struct header_masks {
    static constexpr uint32_t bxmatch = 0x00ff << header_shifts::bxmatch;
    static constexpr uint32_t mAcount = 0x000f << header_shifts::mAcount;
    static constexpr uint32_t orbitmatch = 0x00ff << header_shifts::orbitmatch;
    static constexpr uint32_t warningTestEnabled = 0x0001 << header_shifts::warningTestEnabled;
    static constexpr uint32_t mBcount = 0x000f << header_shifts::mBcount;
    static constexpr uint32_t sBmtfCount = 0x000f << header_shifts::sBmtfCount;
  };

}  // namespace l1ScoutingRun3
#endif  // L1ScoutingRawToDigi_masks_h
