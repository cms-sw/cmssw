#ifndef DATAFORMATS_TCDS_TCDSRAW_H
#define DATAFORMATS_TCDS_TCDSRAW_H

//---------------------------------------------------------------------------
//!  \class TCDSRaw
//!  \brief Structure of raw data from TCDS FED 1014
//!
//!  \author Remi Mommsen - Fermilab
//---------------------------------------------------------------------------

#include <stdint.h>
#include "EventFilter/FEDInterface/interface/fed_header.h"
#include "EventFilter/FEDInterface/interface/fed_trailer.h"

#pragma pack(push)
#pragma pack(1)

namespace tcds {

  struct Sizes_v1
  {
    const unsigned char headerSize;
    const unsigned char summarySize;
    const unsigned char L1AhistSize;
    const unsigned char BSTSize;
    const unsigned char reserved0;
    const unsigned char reserved1;
    const unsigned char reserved2;
    const unsigned char BGOSize;
  };

  struct Header_v1
  {
    uint64_t macAddress;
    uint32_t swVersion;
    uint32_t fwVersion;
    uint32_t reserved0;
    uint32_t recordVersion;
    uint32_t runNumber;
    uint32_t reserved1;
    uint32_t activePartitions2;
    uint32_t bstReceptionStatus;
    uint32_t activePartitions0;
    uint32_t activePartitions1;
    uint32_t nibble;
    uint32_t lumiSection;
    uint16_t nibblesPerLumiSection;
    uint16_t triggerTypeFlags;
    uint16_t reserved5;
    uint16_t inputs;
    uint16_t bxid;
    uint16_t orbitLow;
    uint32_t orbitHigh;
    uint64_t triggerCount;
    uint64_t eventNumber;
  };

  struct L1aInfo_v1
  {
    uint32_t orbitlow;
    uint16_t orbithigh;
    unsigned char reserved2;
    unsigned char ind0;
    uint16_t bxid;
    uint16_t reserved0;
    uint16_t reserved1;
    unsigned char eventtype;
    unsigned char ind1;
  };

  const uint8_t l1aHistoryDepth_v1 = 16;
  struct L1aHistory_v1
  {
    L1aInfo_v1 l1aInfo[l1aHistoryDepth_v1];
  };

  struct BST_v1
  {
    uint32_t gpstimelow;
    uint32_t gpstimehigh;
    uint32_t bireserved8_11;
    uint32_t bireserved12_15;
    uint16_t bstMaster;
    uint16_t turnCountLow;
    uint16_t turnCountHigh;
    uint16_t lhcFillLow;
    uint16_t lhcFillHigh;
    uint16_t beamMode;
    uint16_t particleTypes;
    uint16_t beamMomentum;
    uint32_t intensityBeam1;
    uint32_t intensityBeam2;
    uint32_t bireserved40_43;
    uint32_t bireserved44_47;
    uint32_t bireserved48_51;
    uint32_t bireserved52_55;
    uint32_t bireserved56_59;
    uint32_t bireserved60_63;
  };

  struct LastBGo_v1
  {
    uint32_t orbitlow;
    uint16_t orbithigh;
    uint16_t reserved;
  };

  const uint8_t bgoCount_v1 = 64;
  struct BGoHistory_v1
  {
    uint64_t bgoHistoryHeader;
    struct LastBGo_v1 lastBGo[bgoCount_v1];
  };

  struct Raw_v1
  {
    fedh_t               fedHeader;
    struct Sizes_v1      sizes;
    struct Header_v1     header;
    struct L1aHistory_v1 l1aHistory;
    struct BST_v1        bst;
    struct BGoHistory_v1 bgoHistory;
    fedt_t               fedTrailer;
  };

}

#pragma pack(pop)

#endif // DATAFORMATS_TCDS_TCDSRAW_H
