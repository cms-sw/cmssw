#ifndef DATAFORMATS_BTLELECTRONICSMAPPING_H
#define DATAFORMATS_BTLELECTRONICSMAPPING_H 1

#include <ostream>
#include <cstdint>

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include <Geometry/MTDCommonData/interface/MTDTopologyMode.h>

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
  BTLElectronicsMapping(const BTLDetId::CrysLayout lay);

  // Get SiPM Channel number from crystal
  int SiPMCh(uint32_t smodCopy, uint32_t crystal, uint32_t SiPMSide) const;
  int SiPMCh(BTLDetId det, uint32_t SiPMSide) const;
  int SiPMCh(uint32_t rawID, uint32_t SiPMSide) const;

  SiPMChPair GetSiPMChPair(uint32_t smodCopy, uint32_t crystal) const;
  SiPMChPair GetSiPMChPair(BTLDetId det) const;
  SiPMChPair GetSiPMChPair(uint32_t rawID) const;

  // Get TOFHIR Channel number from crystal
  int TOFHIRCh(uint32_t smodCopy, uint32_t crystal, uint32_t SiPMSide) const;
  int TOFHIRCh(BTLDetId det, uint32_t SiPMSide) const;
  int TOFHIRCh(uint32_t rawID, uint32_t SiPMSide) const;

  TOFHIRChPair GetTOFHIRChPair(uint32_t smodCopy, uint32_t crystal) const;
  TOFHIRChPair GetTOFHIRChPair(BTLDetId det) const;
  TOFHIRChPair GetTOFHIRChPair(uint32_t rawID) const;

  // Get xtal from TOFHIR Channel number
  int THChToXtal(uint32_t smodCopy, uint32_t THCh) const;
  BTLDetId THChToBTLDetId(
      uint32_t zside, uint32_t rod, uint32_t runit, uint32_t dmodule, uint32_t smodCopy, uint32_t THCh) const;

  /** Returns TOFHIR ASIC number in construction database. */
  int TOFHIRASIC(uint32_t dmodule, uint32_t smodCopy) const;
  int TOFHIRASIC(BTLDetId det) const;
  int TOFHIRASIC(uint32_t rawID) const;

  /** Returns FE board number */
  int FEBoardFromDM(uint32_t dmodule) const;
  int FEBoard(BTLDetId det) const;
  int FEBoard(uint32_t rawID) const;

  /** Returns CC board number */
  int CCBoardFromRU(uint32_t dmodule) const;
  int CCBoard(BTLDetId det) const;
  int CCBoard(uint32_t rawID) const;

  /** ======== DAQ cabling mapping ========== **/
  /** E-LINK  <-> TOFHIR/SM **/
  /** HS-link <-> CC/RU to Serenity **/
  /** S-link <-> FED Id (6 trays) **/
  /** ======================================= **/

  /** E-LINK **/
  /** E-LINK identifies which TOFHIR/SM within a CC. The mapping depends if lpgbt L0 or L1 is used. Default is L0. **/
  /** Ref: https://gitlab.cern.ch/btl-commissioning/mtd_daq/-/blob/tif_new_s1p2/src/mtd_daq/configurator/mappings.py?ref_type=heads#L79-130 */

  static constexpr std::array<std::pair<int, int>, BTLDetId::kDModulesPerRU * 2> ELINK_to_FE_mapping_L0{{
      {7, 1},  // e-link 0:  {DM 7, chipId 1}
      {7, 0},  // e-link 1:  {DM 7, chipId 0}
      {3, 1}, {3, 0}, {2, 1}, {2, 0}, {0, 0},  {0, 1},  {1, 1},  {1, 0},  {5, 0}, {5, 1}, {4, 1}, {4, 0},
      {8, 0}, {9, 1}, {9, 0}, {8, 1}, {10, 0}, {10, 1}, {11, 1}, {11, 0}, {6, 1}, {6, 0}  // e-link 23:  {DM 6, chipId 0}
  }};

  static constexpr std::array<std::pair<int, int>, BTLDetId::kDModulesPerRU * 2> ELINK_to_FE_mapping_L1{{
      {6, 1},  // e-link 0:  {DM 6, chipId 1}
      {3, 1},  // e-link 1:  {DM 3, chipId 1}
      {11, 0}, {11, 1}, {3, 0}, {6, 0}, {2, 1}, {2, 0}, {1, 0},  {1, 1},  {0, 0},  {0, 1}, {5, 1}, {5, 0},
      {4, 0},  {4, 1},  {9, 1}, {9, 0}, {8, 0}, {8, 1}, {10, 0}, {10, 1}, {11, 0}, {11, 1}  // e-link 23:  {DM 11, chipId 1}
  }};

  // ELINK Reverse mapping: given an FE/DM (0-11) and a ChipId (0-1), find the corresponding e-link
  static constexpr auto buildFEtoElinkMapping = [](auto const& mapping) {
    std::array<std::array<int, 2>, BTLDetId::kDModulesPerRU> tmp{};  // pair is (DM, chipId), there are 12 pairs.
    for (auto& row : tmp)
      row.fill(-1);  // initialize to -1
    for (unsigned int elink = 0; elink < BTLDetId::kDModulesPerRU * 2; ++elink) {
      tmp[mapping[elink].first][mapping[elink].second] = static_cast<int>(elink);
    }
    return tmp;
  };

  static constexpr auto FE_to_ELINK_mapping_L0 = buildFEtoElinkMapping(ELINK_to_FE_mapping_L0);
  static constexpr auto FE_to_ELINK_mapping_L1 = buildFEtoElinkMapping(ELINK_to_FE_mapping_L1);

  // -- Get the e-link for a given SM
  int elinkFromSM(uint32_t dmodule, uint32_t smodCopy, int lpgbt_id = 0) const;  // e-link from DM and SM
  int elink(BTLDetId det, int lpgbt_id = 0) const;                               // e-link from BTLDetId
  int elink(uint32_t rawID, int lpgbt_id = 0) const;                             // e-link from rawID
  // -- Get the (DM,SM) for a given e-link
  std::pair<int, int> elinkToSM(int elink, int lpgbt_id = 0) const;

  // ** HS-LINK **/
  // ** HS-link = link CC/RU to Serenity **/
  // Mapping of FPGA Tx port number (hs-link) to optical Tx channels provided by O. Sahin.
  // Note: this mapping is not strictly sequential!!!
  // In totale there are 72 = 6 x 12 hs-links (from 4 to 75) in blocks of 12. Each block will correspond to one tray.
  // Within each block:
  // Opt Tx 0, 2, 4, 6, 8, 10 --> correspond to CC 0,1,2,3,4,5 for LPGBT0
  // Opt Tx 1, 3, 5, 7, 9, 11 --> correspond to CC 0,1,2,3,4,5 for LPGBT1
  // !!!!  Which tray goes to which block for now is arbitrary, but for the fact that 6 trays are contiguos
  // TEMPORARY MAPPING: within a group of 6 trays ( = supertray): tray 0 --> first block of 12 links, tray 1--> second block of 12 links, etc.
  // For the first block (FF_N5), channel ids of the optical tx are reversed.

  /*  struct Tx {
    const char* name;
    int index;
  };
       
 static constexpr std::array<Tx, 76> Tx_map = {{
    {}, {}, {}, {},  // 0-3 unused

    // FF_N5_tx (4-15)
    { "FF_N5_tx", 1 },   // 4
    { "FF_N5_tx", 3 },   // 5
    { "FF_N5_tx", 5 },   // 6
    { "FF_N5_tx", 0 },   // 7
    { "FF_N5_tx", 2 },   // 8
    { "FF_N5_tx", 4 },   // 9
    { "FF_N5_tx", 6 },   // 10
    { "FF_N5_tx", 8 },   // 11
    { "FF_N5_tx", 10 },  // 12
    { "FF_N5_tx", 7 },   // 13
    { "FF_N5_tx", 9 },   // 14
    { "FF_N5_tx", 11 },  // 15

    // FF_S0_tx (16 - 27)
    { "FF_S0_tx", 11 },  // 16
    { "FF_S0_tx", 9 },   // 17
    { "FF_S0_tx", 7 },   // 18
    { "FF_S0_tx", 10 },  // 19
    { "FF_S0_tx", 8 },   // 20
    { "FF_S0_tx", 6 },   // 21
    { "FF_S0_tx", 4 },   // 22
    { "FF_S0_tx", 2 },   // 23
    { "FF_S0_tx", 0 },   // 24
    { "FF_S0_tx", 5 },   // 25
    { "FF_S0_tx", 3 },   // 26
    { "FF_S0_tx", 1 },   // 27

    // FF_S1_tx (28 - 39)
    { "FF_S1_tx", 11 },  // 28
    { "FF_S1_tx", 9 },   // 29
    { "FF_S1_tx", 7 },   // 30
    { "FF_S1_tx", 10 },  // 31
    { "FF_S1_tx", 8 },   // 32
    { "FF_S1_tx", 6 },   // 33
    { "FF_S1_tx", 4 },   // 34
    { "FF_S1_tx", 2 },   // 35
    { "FF_S1_tx", 0 },   // 36
    { "FF_S1_tx", 5 },   // 37
    { "FF_S1_tx", 3 },   // 38
    { "FF_S1_tx", 1 },   // 39

    // FF_S2_tx (40 - 51)
    { "FF_S2_tx", 11 },  // 40
    { "FF_S2_tx", 9 },   // 41
    { "FF_S2_tx", 7 },   // 42
    { "FF_S2_tx", 10 },  // 43
    { "FF_S2_tx", 8 },   // 44
    { "FF_S2_tx", 6 },   // 45
    { "FF_S2_tx", 4 },   // 46
    { "FF_S2_tx", 2 },   // 47
    { "FF_S2_tx", 0 },   // 48
    { "FF_S2_tx", 5 },   // 49
    { "FF_S2_tx", 3 },   // 50
    { "FF_S2_tx", 1 },   // 51

    // FF_S3_tx (52 - 63)
    { "FF_S3_tx", 11 },  // 52
    { "FF_S3_tx", 9 },   // 53
    { "FF_S3_tx", 7 },   // 54
    { "FF_S3_tx", 10 },  // 55
    { "FF_S3_tx", 8 },   // 56
    { "FF_S3_tx", 6 },   // 57
    { "FF_S3_tx", 4 },   // 58
    { "FF_S3_tx", 2 },   // 59
    { "FF_S3_tx", 0 },   // 60
    { "FF_S3_tx", 5 },   // 61
    { "FF_S3_tx", 3 },   // 62
    { "FF_S3_tx", 1 },   // 63

    // FF_N4_tx (64 - 75)
    { "FF_N4_tx", 11 },  // 64
    { "FF_N4_tx", 9 },   // 65
    { "FF_N4_tx", 7 },   // 66
    { "FF_N4_tx", 10 },  // 67
    { "FF_N4_tx", 8 },   // 68
    { "FF_N4_tx", 6 },   // 69
    { "FF_N4_tx", 4 },   // 70
    { "FF_N4_tx", 2 },   // 71
    { "FF_N4_tx", 0 },   // 72
    { "FF_N4_tx", 5 },   // 73
    { "FF_N4_tx", 3 },   // 74
    { "FF_N4_tx", 1 }    // 75
						}};
       
  */

  static constexpr uint32_t kNumHSLinks = 72;     // number of HS links: in each Serenity 6 trays x 12 links
  static constexpr uint32_t kOffsetHSLinks = 4;   // offset (HS link Ids start from 4 (0-3 reserved)
  static constexpr uint32_t MIN_SLINK_ID = 1000;  // arbitrary for now

  // Define an array of 12 elements, each element is the optical Tx channel Id (this depends on the FF). Then each channel will map to a CC/RU
  static constexpr std::array<int, 2 * BTLDetId::kRUPerRod> optTxCh_n5 = {
      1, 3, 5, 0, 2, 4, 6, 8, 10, 7, 9, 11};  // 12 = 6 RUs x 2 LPGBTs
  static constexpr std::array<int, 2 * BTLDetId::kRUPerRod> optTxCh_common = {
      11, 9, 7, 10, 8, 6, 4, 2, 0, 5, 3, 1};  // 12 = 6 RUs x 2 LPGBTs

  static constexpr auto OptTx_map = []() {
    std::array<int, kNumHSLinks + kOffsetHSLinks> tmp;
    tmp.fill(-1);
    for (unsigned int i = 0; i < kNumHSLinks / 12; i++) {
      for (int j = 0; j < 12; j++) {
        int hslink_id = kOffsetHSLinks + i * 12 + j;
        tmp[hslink_id] = (i == 0) ? optTxCh_n5[j] : optTxCh_common[j];
      }
    }
    return tmp;
  }();

  // Build inverse mapping
  static constexpr auto tx_to_index = [](const std::array<int, 12>& p) {
    std::array<int, 12> inv{};
    inv.fill(-1);
    for (int i = 0; i < 12; ++i)
      inv[p[i]] = i;
    return inv;
  };
  static constexpr auto tx_inv_n5 = tx_to_index(optTxCh_n5);
  static constexpr auto tx_inv_common = tx_to_index(optTxCh_common);

  // -- Get the CC/RU corresponding to a given HS-link
  int hslinkToRU(int hslink) const;
  // -- Get the HS-link corresponding to a given RU/CC in a tray
  int hslinkFromRU(uint32_t runit, uint32_t tray, int lpgbt_id = 0) const;
  int hslink(BTLDetId det, int lpgbt_id = 0) const;
  int hslink(uint32_t rawID, int lpgbt_id = 0) const;

  /** S-link **/
  /** one S-link corresponds to a group of 6 trays. one S-link = one FEDId **/
  /** !!!!! TEMPORARY mapping for now until FEDId not assigned !!!!! **/
  // TEMPORARY MAPPING:                                                                                                                                                                                         // trays [0-35], z- --> Slinks [0,5]
  // trays [0-35], z+ --> Slinks [6,11]

  // -- Get the tray Id from combination of S-link (--> group fo 6 trays) and HS-link (which tray in that S-link, i.e.
  std::pair<uint32_t, uint32_t> getTrayFromLinks(
      int slink, int hslink) const;  // return tray number [0-35] and z side [0,1] given S-link and HS-link Id
  // -- Get the S-link for a given tray
  int SlinkFromTray(uint32_t tray, uint32_t zside) const;
  int Slink(BTLDetId det) const;
  int Slink(uint32_t rawID) const;

private:
};

#endif
