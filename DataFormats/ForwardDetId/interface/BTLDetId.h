#ifndef DataFormats_BTLDetId_BTLDetId_h
#define DataFormats_BTLDetId_BTLDetId_h

#include "DataFormats/ForwardDetId/interface/MTDDetId.h"
#include <ostream>
#include <array>

/** 
    @class BTLDetId
    @brief Detector identifier class for the Barrel Timing Layer.
    The crystal count must start from 0, copy number must be scaled by 1 unit.

    bit 15-10: module sequential number
    bit 9-8  : crystal type (1 - 3)
    bit 7-6  : readout unit sequential number within a type ( 1 - 2 )
    bit 5-0  : crystal sequential number within a module ( 0 - 15 )
*/

class BTLDetId : public MTDDetId {
public:
  static constexpr uint32_t kBTLmoduleOffset = 10;
  static constexpr uint32_t kBTLmoduleMask = 0x3F;
  static constexpr uint32_t kBTLmodTypeOffset = 8;
  static constexpr uint32_t kBTLmodTypeMask = 0x3;
  static constexpr uint32_t kBTLRUOffset = 6;
  static constexpr uint32_t kBTLRUMask = 0x3;
  static constexpr uint32_t kBTLCrystalOffset = 0;
  static constexpr uint32_t kBTLCrystalMask = 0x3F;

  /// range constants, need two sets for the time being (one for tiles and one for bars)
  static constexpr uint32_t HALF_ROD = 36;
  static constexpr uint32_t kModulesPerRODBarPhiFlat = 48;
  static constexpr uint32_t kModulePerTypeBarPhiFlat = 48 / 3;
  static constexpr uint32_t kRUPerTypeV2 = 2;
  static constexpr uint32_t kModulesPerRUV2 = 24;
  static constexpr uint32_t kCrystalsPerModuleV2 = 16;
  static constexpr uint32_t kModulesPerTrkV2 = 3;
  static constexpr uint32_t kCrystalTypes = 3;

  // Number of crystals in BTL according to TDR design, valid also for barphiflat scenario:
  // 16 crystals x 24 modules x 2 readout units/type x 3 types x 36 rods/side x 2 sides
  //
  static constexpr uint32_t kCrystalsBTL =
      kCrystalsPerModuleV2 * kModulesPerRUV2 * kRUPerTypeV2 * kCrystalTypes * HALF_ROD * 2;

  enum class CrysLayout { tile = 1, bar = 2, barzflat = 3, barphiflat = 4, v2 = 5 };

  // ---------- Constructors, enumerated types ----------

  /** Construct a null id */
  BTLDetId() : MTDDetId(DetId::Forward, ForwardSubdetector::FastTime) {
    id_ |= (MTDType::BTL & kMTDsubdMask) << kMTDsubdOffset;
  }

  /** Construct from a raw value */
  BTLDetId(const uint32_t& raw_id) : MTDDetId(raw_id) { ; }

  /** Construct from generic DetId */
  BTLDetId(const DetId& det_id) : MTDDetId(det_id.rawId()) { ; }

  /** Construct from complete geometry information, v1 **/
  BTLDetId(uint32_t zside, uint32_t rod, uint32_t module, uint32_t modtyp, uint32_t crystal)
      : MTDDetId(DetId::Forward, ForwardSubdetector::FastTime) {
    id_ |= (MTDType::BTL & kMTDsubdMask) << kMTDsubdOffset | (zside & kZsideMask) << kZsideOffset |
           (rod & kRodRingMask) << kRodRingOffset | (module & kBTLmoduleMask) << kBTLmoduleOffset |
           (modtyp & kBTLmodTypeMask) << kBTLmodTypeOffset | ((crystal - 1) & kBTLCrystalMask) << kBTLCrystalOffset;
  }

  /** Construct from complete geometry information, v2 **/
  BTLDetId(uint32_t zside, uint32_t rod, uint32_t runit, uint32_t module, uint32_t modtyp, uint32_t crystal)
      : MTDDetId(DetId::Forward, ForwardSubdetector::FastTime) {
    id_ |= (MTDType::BTL & kMTDsubdMask) << kMTDsubdOffset | (zside & kZsideMask) << kZsideOffset |
           (rod & kRodRingMask) << kRodRingOffset | (module & kBTLmoduleMask) << kBTLmoduleOffset |
           (modtyp & kBTLmodTypeMask) << kBTLmodTypeOffset | (runit & kBTLRUMask) << kBTLRUOffset |
           ((crystal - 1) & kBTLCrystalMask) << kBTLCrystalOffset;
  }

  // ---------- Common methods ----------

  /** Returns BTL module number. */
  inline int module() const { return (id_ >> kBTLmoduleOffset) & kBTLmoduleMask; }

  /** Returns BTL crystal type number. */
  inline int modType() const { return (id_ >> kBTLmodTypeOffset) & kBTLmodTypeMask; }

  /** Returns BTL crystal number. */
  inline int crystal() const { return ((id_ >> kBTLCrystalOffset) & kBTLCrystalMask) + 1; }

  /** Returns BTL readout unit number per type. */
  inline int runit() const { return (id_ >> kBTLRUOffset) & kBTLRUMask; }

  /** return the row in GeomDet language **/
  inline int row(unsigned nrows = kCrystalsPerModuleV2) const {
    return (crystal() - 1) % nrows;  // anything else for now
  }

  /** return the column in GeomDetLanguage **/
  inline int column(unsigned nrows = kCrystalsPerModuleV2) const { return (crystal() - 1) / nrows; }

  /** create a Geographical DetId for Tracking **/
  BTLDetId geographicalId(CrysLayout lay) const;
};

std::ostream& operator<<(std::ostream&, const BTLDetId&);

#endif  // DataFormats_BTLDetId_BTLDetId_h
