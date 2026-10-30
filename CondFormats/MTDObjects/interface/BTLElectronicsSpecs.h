#ifndef CondFormats_MTDObjects_BTLElectronicsSpecs_h
#define CondFormats_MTDObjects_BTLElectronicsSpecs_h

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
  // DAQ constants
  // ============================================================
  static constexpr int32_t kNumberOfFEDs = 144;    // 72 trays x 2 Slinks for each tray (or 12 Slinks for each Serenity)
  static constexpr int32_t kNumberOfSlinks = 144;  // 72 trays x 2 Slinks for each tray (or 12 Slinks for each Serenity)
  static constexpr int32_t kFirstSlinkId = 0;      // arbitrary for now
  static constexpr int32_t kFirstFEDId = 0;        // arbitrary for now

  static constexpr int32_t nSLinksPerSerenity = 12;  // number of S-links per Serenity board
  static constexpr int32_t nFEDsPerSerenity = 12;    // 1 S-link  = 1 FED ID
  static constexpr int32_t nHSLinksPerFED = 3;       // number of HS-links per Slink/FED (3 RUs --> 1 Slink)
  static constexpr int32_t nHSLinksPerSerenity = nHSLinksPerFED * nFEDsPerSerenity;  // = 36
  static constexpr int32_t nELinksPerHSLink = 24;   // number of elinks (chips) per HS-link
  static constexpr int32_t nChannelsPerELink = 32;  // channels per TOFHIR2C chip

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

  // Table of TOFHIR channel partners.
  // The array index is the TOFHIR channelId.
  // The value is the channelId of its partner.
  // !!!!!!!!!! DA RICONTROLLARE  !!!!
  static constexpr std::array<int8_t, 32> kTofhirChannelPartner = {24, 25, 31, 30, 29, 26, 28, 27, 23, 22, 19,
                                                                   21, 20, 17, 18, 16, 15, 13, 14, 10, 12, 11,
                                                                   9,  8,  0,  1,  5,  7,  6,  4,  3,  2};

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
      {7, 0}, {7, 1}, {3, 0}, {6, 0}, {2, 1}, {2, 0}, {1, 0},  {1, 1},  {0, 0},  {0, 1}, {5, 1}, {5, 0},
      {4, 0}, {4, 1}, {9, 1}, {9, 0}, {8, 0}, {8, 1}, {10, 0}, {10, 1}, {11, 0}, {11, 1}  // e-link 23:  {DM 11, chipId 1}
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

}  // namespace BTLElectronicsSpecs

#endif
