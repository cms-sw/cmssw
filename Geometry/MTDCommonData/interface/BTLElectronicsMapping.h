#ifndef DATAFORMATS_BTLELECTRONICSMAPPING_H
#define DATAFORMATS_BTLELECTRONICSMAPPING_H 1

#include <ostream>
#include <cstdint>

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

/** \brief BTL TOFHIR channel mapping with crystal BTLDetId
    Convention:
      SiPMside 0 == Minus Side
      SiPMside 1 == Plus Side
 */

class BTLElectronicsMapping {
public:
  struct SiPMChPair {
    int Minus;
    int Plus;
  };

  struct TOFHIRChPair {
    int Minus;
    int Plus;
  };

  // Map SiPM Channel to crystal bars for Forward module orientation
  static constexpr std::array<uint32_t, BTLDetId::kCrystalsPerModuleV2 * 2> SiPMChannelMapFW{
      {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
       15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0}};
  // Map SiPM Channel to crystal bars for Backward module orientation
  static constexpr std::array<uint32_t, BTLDetId::kCrystalsPerModuleV2 * 2> SiPMChannelMapBW{
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
       31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16}};

  // Map TOFHIR Channel to SiPM Channel
  static constexpr std::array<uint32_t, BTLDetId::kCrystalsPerModuleV2 * 2> THChannelMap{
      {4,  1,  0,  3,  2,  6,  7,  9,  5,  11, 8,  12, 10, 14, 15, 13,
       17, 16, 18, 19, 20, 23, 21, 26, 22, 27, 28, 31, 30, 24, 25, 29}};

  /** Default constructor -- invalid value */
  BTLElectronicsMapping();

  // Get SiPM Channel number from crystal
  int SiPMCh(uint32_t smodCopy, uint32_t crystal, uint32_t SiPMSide);
  int SiPMCh(BTLDetId det, uint32_t SiPMSide);
  int SiPMCh(uint32_t rawID, uint32_t SiPMSide);

  SiPMChPair GetSiPMChPair(uint32_t smodCopy, uint32_t crystal);
  SiPMChPair GetSiPMChPair(BTLDetId det);
  SiPMChPair GetSiPMChPair(uint32_t rawID);

  // Get TOFHIR Channel number from crystal
  int TOFHIRCh(uint32_t smodCopy, uint32_t crystal, uint32_t SiPMSide);
  int TOFHIRCh(BTLDetId det, uint32_t SiPMSide);
  int TOFHIRCh(uint32_t rawID, uint32_t SiPMSide);

  TOFHIRChPair GetTOFHIRChPair(uint32_t smodCopy, uint32_t crystal);
  TOFHIRChPair GetTOFHIRChPair(BTLDetId det);
  TOFHIRChPair GetTOFHIRChPair(uint32_t rawID);

  // Get xtal from TOFHIR Channel number
  int THChToXtal(uint32_t smodCopy, uint32_t THCh);
  BTLDetId THChToBTLDetId(
      uint32_t zside, uint32_t rod, uint32_t runit, uint32_t dmodule, uint32_t smodCopy, uint32_t THCh);

  /** Returns TOFHIR ASIC number in construction database. */
  int TOFHIRASIC(uint32_t dmodule, uint32_t smodCopy);
  int TOFHIRASIC(BTLDetId det);
  int TOFHIRASIC(uint32_t rawID);

  /** Returns FE board number */
  int FEBoardFromDM(uint32_t dmodule);
  int FEBoard(BTLDetId det);
  int FEBoard(uint32_t rawID);

private:
};

#endif
