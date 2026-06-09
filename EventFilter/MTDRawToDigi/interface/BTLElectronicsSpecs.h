#ifndef EventFilter_MTDRawToDigi_BTLElectronicsSpecs_h
#define EventFilter_MTDRawToDigi_BTLElectronicsSpecs_h

#include <array>
#include <utility>
#include <cstdint>

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

/**
 *  Lookup tables - Pure firmware / DAQ encoding specifications.
 *
 */

namespace BTLElectronicsSpecs {

  // ============================================================
  // TOFHIR channel ID mapping
  // ============================================================

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

  // ============================================================
  // E-LINK mapping
  // ============================================================
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

  // ============================================================
  // HS-LINK mapping
  // ============================================================
  // ** HS-link = link CC/RU to Serenity **/
  // Mapping of FPGA Tx port number (= HS-link) to optical Tx channels provided by O. Sahin.
  // Note: this mapping is not strictly sequential
  // There are 72 = 6 x 12 hs-links (now from 4 to 75) in blocks of 12. Each block will correspond to one tray.
  // Within each block:
  // Opt Tx 0, 2, 4, 6, 8, 10 --> correspond to CC 0,1,2,3,4,5 for LPGBT0
  // Opt Tx 1, 3, 5, 7, 9, 11 --> correspond to CC 0,1,2,3,4,5 for LPGBT1

  // Define an array of 12 elements, each element is the optical Tx channel Id (this depends on the FF).
  // For the first block (FF_N5), channel ids of the optical tx are reversed.
  // Each optical Tx channel Id maps to a CC/RU
  static constexpr std::array<int, 2 * BTLDetId::kRUPerRod> optTxCh_n5 = {
      1, 3, 5, 0, 2, 4, 6, 8, 10, 7, 9, 11};  // 12 = 6 RUs x 2 LPGBTs
  static constexpr std::array<int, 2 * BTLDetId::kRUPerRod> optTxCh_common = {
      11, 9, 7, 10, 8, 6, 4, 2, 0, 5, 3, 1};  // 12 = 6 RUs x 2 LPGBTs

  // ============================================================
  // DAQ constants
  // ============================================================
  static constexpr uint32_t kHSLinksNum = 72;    // number of HS links: in each Serenity 6 trays x 12 links
  static constexpr uint32_t kHSLinksOffset = 4;  // offset (HS link Ids start from 4 (0-3 reserved)
  static constexpr uint32_t kFirstFEDId = 0;     // arbitrary for now

  static constexpr auto OptTx_map = []() {
    std::array<int, kHSLinksNum + kHSLinksOffset> tmp;
    tmp.fill(-1);
    for (unsigned int i = 0; i < kHSLinksNum / 12; i++) {
      for (int j = 0; j < 12; j++) {
        int hslink_id = kHSLinksOffset + i * 12 + j;
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

}  // namespace BTLElectronicsSpecs

#endif
