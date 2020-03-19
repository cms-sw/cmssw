#ifndef DATAFORMATS_TCDS_TCDSRAW_H
#define DATAFORMATS_TCDS_TCDSRAW_H

//---------------------------------------------------------------------------
//!  \class TCDSRaw
//!  \brief Structure of raw data from TCDS FED 1014
//!
//!  \author Remi Mommsen - Fermilab
//---------------------------------------------------------------------------

#include <cstdint>

#pragma pack(push)
#pragma pack(1)

namespace tcds {

  struct Sizes_v1 {
    const uint8_t headerSize;
    const uint8_t summarySize;
    const uint8_t L1AhistSize;
    const uint8_t BSTSize;
    const uint8_t reserved0;
    const uint8_t reserved1;
    const uint8_t reserved2;
    const uint8_t BGOSize;
  };

  struct Header_v1 {
    const uint64_t macAddress;
    const uint32_t swVersion;
    const uint32_t fwVersion;
    const uint32_t reserved0;
    const uint32_t recordVersion;
    const uint32_t runNumber;
    const uint32_t reserved1;
    const uint32_t activePartitions2;
    const uint32_t bstReceptionStatus;
    const uint32_t activePartitions0;
    const uint32_t activePartitions1;
    const uint32_t nibble;
    const uint32_t lumiSection;
    const uint16_t nibblesPerLumiSection;
    const uint16_t triggerTypeFlags;
    const uint16_t reserved5;
    const uint16_t inputs;
    const uint16_t bxid;
    const uint16_t orbitLow;
    const uint32_t orbitHigh;
    const uint64_t triggerCount;
    const uint64_t eventNumber;
  };

  struct L1aInfo_v1 {
    const uint32_t orbitlow;
    const uint16_t orbithigh;
    const uint8_t reserved2;
    const uint8_t ind0;
    const uint16_t bxid;
    const uint16_t reserved0;
    const uint16_t reserved1;
    const uint8_t eventtype;
    const uint8_t ind1;
  };

  const uint8_t l1aHistoryDepth_v1 = 16;
  struct L1aHistory_v1 {
    L1aInfo_v1 l1aInfo[l1aHistoryDepth_v1];
  };

  struct BST_v1 {
    const uint32_t gpstimelow;
    const uint32_t gpstimehigh;
    const uint32_t bireserved8_11;
    const uint32_t bireserved12_15;
    const uint16_t bstMaster;
    const uint16_t turnCountLow;
    const uint16_t turnCountHigh;
    const uint16_t lhcFillLow;
    const uint16_t lhcFillHigh;
    const uint16_t beamMode;
    const uint16_t particleTypes;
    const uint16_t beamMomentum;
    const uint32_t intensityBeam1;
    const uint32_t intensityBeam2;
    const uint32_t bireserved40_43;
    const uint32_t bireserved44_47;
    const uint32_t bireserved48_51;
    const uint32_t bireserved52_55;
    const uint32_t bireserved56_59;
    const uint32_t bireserved60_63;
  };

  struct LastBGo_v1 {
    const uint32_t orbitlow;
    const uint16_t orbithigh;
    const uint16_t reserved;
  };

  const uint8_t bgoCount_v1 = 64;
  struct BGoHistory_v1 {
    const uint64_t bgoHistoryHeader;
    const struct LastBGo_v1 lastBGo[bgoCount_v1];
  };

  struct Raw_v1 {
    const struct Sizes_v1 sizes;
    const struct Header_v1 header;
    const struct L1aHistory_v1 l1aHistory;
    const struct BST_v1 bst;
    const struct BGoHistory_v1 bgoHistory;
  };

}  // namespace tcds

#pragma pack(pop)

#endif  // DATAFORMATS_TCDS_TCDSRAW_H
