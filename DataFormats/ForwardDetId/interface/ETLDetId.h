#ifndef DataFormats_ETLDetId_ETLDetId_h
#define DataFormats_ETLDetId_ETLDetId_h

#include "DataFormats/ForwardDetId/interface/MTDDetId.h"
#include <ostream>

/** 
    @class ETLDetId
    @brief Detector identifier class for the Endcap Timing Layer.

    bit 15-5 : module sequential number
    bit 4-3  : module type (unused so far)
    bit 2-1  : sensor
*/

class ETLDetId : public MTDDetId {
private:
  // for conversion from old to new module bit field
  static constexpr uint32_t kETLoldToNewShift = 2;
  static constexpr uint32_t kETLoldFieldMask = 0x7FFF;
  static constexpr uint32_t kETLformatV2 = 1;

public:
  static constexpr uint32_t kETLmoduleOffset = 5;
  static constexpr uint32_t kETLmoduleMask = 0x7FF;
  static constexpr uint32_t kETLmodTypeOffset = 3;
  static constexpr uint32_t kETLmodTypeMask = 0x3;
  static constexpr uint32_t kETLsensorOffset = 1;
  static constexpr uint32_t kETLsensorMask = 0x3;

  static constexpr int kETLv1maxRing = 11;
  static constexpr int kETLv1maxModule = 176;
  static constexpr int kETLv1nDisc = 1;

  /// constants for the TDR ETL model
  static constexpr uint32_t kETLnDiscOffset = 3;
  static constexpr uint32_t kETLnDiscMask = 0x1;
  static constexpr uint32_t kETLdiscSideOffset = 2;
  static constexpr uint32_t kETLdiscSideMask = 0x1;
  static constexpr uint32_t kETLsectorMask = 0x3;

  static constexpr int kETLv4maxRing = 16;
  static constexpr int kETLv4maxSector = 4;
  static constexpr int kETLv4maxModule = 248;
  static constexpr int kETLv4nDisc = 2;

  static constexpr int kETLv5maxRing = 14;
  static constexpr int kETLv5maxSector = 2;
  static constexpr int kETLv5maxModule = 517;
  static constexpr int kETLv5nDisc = kETLv4nDisc;

  static constexpr uint32_t kSoff = 4;

  // ---------- Constructors, enumerated types ----------

  /** Construct a null id */
  ETLDetId() : MTDDetId(DetId::Forward, ForwardSubdetector::FastTime) {
    id_ |= (MTDType::ETL & kMTDsubdMask) << kMTDsubdOffset;
    id_ |= kETLformatV2;
  }

  /** Construct from a raw value */
  ETLDetId(const uint32_t& raw_id) {
    uint32_t tmpId = raw_id;
    if ((tmpId & kETLformatV2) == 0) {
      tmpId = newForm(tmpId);
    }
    id_ = MTDDetId(tmpId).rawId();
  }

  /** Construct from generic DetId */
  ETLDetId(const DetId& det_id) {
    uint32_t tmpId = det_id.rawId();
    if ((tmpId & kETLformatV2) == 0) {
      tmpId = newForm(tmpId);
    }
    id_ = MTDDetId(tmpId).rawId();
  }

  /** Construct and fill only the det and sub-det fields. */
  // pre v8
  ETLDetId(uint32_t zside, uint32_t ring, uint32_t module, uint32_t modtyp)
      : MTDDetId(DetId::Forward, ForwardSubdetector::FastTime) {
    id_ |= (MTDType::ETL & kMTDsubdMask) << kMTDsubdOffset | (zside & kZsideMask) << kZsideOffset |
           (ring & kRodRingMask) << kRodRingOffset | (module & kETLmoduleMask) << kETLmoduleOffset |
           (modtyp & kETLmodTypeMask) << kETLmodTypeOffset;
    id_ |= kETLformatV2;
  }
  // v8
  ETLDetId(uint32_t zside, uint32_t ring, uint32_t module, uint32_t modtyp, uint32_t sensor)
      : MTDDetId(DetId::Forward, ForwardSubdetector::FastTime) {
    id_ |= (MTDType::ETL & kMTDsubdMask) << kMTDsubdOffset | (zside & kZsideMask) << kZsideOffset |
           (ring & kRodRingMask) << kRodRingOffset | (module & kETLmoduleMask) << kETLmoduleOffset |
           (modtyp & kETLmodTypeMask) << kETLmodTypeOffset | (sensor & kETLsensorMask) << kETLsensorOffset;
    id_ |= kETLformatV2;
  }

  /** ETL TDR Construct and fill only the det and sub-det fields. */
  /** input disc runs from 0 to 1 */

  inline uint32_t encodeSector(uint32_t& disc, uint32_t& discside, uint32_t& sector) const {
    return (sector + discside * kSoff + 2 * kSoff * disc);
  }

  /** decode encoded "ring" field, disc numbered from 1 to 2, as in dedicated method */

  static void decodeSector(const uint32_t rr, uint32_t& nDisc, uint32_t& discSide, uint32_t& sector) {
    nDisc = (((rr - 1) >> kETLnDiscOffset) & kETLnDiscMask) + 1;
    discSide = ((rr - 1) >> kETLdiscSideOffset) & kETLdiscSideMask;
    sector = ((rr - 1) & kETLsectorMask) + 1;
  }

  // pre v8
  ETLDetId(uint32_t zside, uint32_t disc, uint32_t discside, uint32_t sector, uint32_t module, uint32_t modtyp)
      : MTDDetId(DetId::Forward, ForwardSubdetector::FastTime) {
    id_ |= (MTDType::ETL & kMTDsubdMask) << kMTDsubdOffset | (zside & kZsideMask) << kZsideOffset |
           (encodeSector(disc, discside, sector) & kRodRingMask) << kRodRingOffset |
           (module & kETLmoduleMask) << kETLmoduleOffset | (modtyp & kETLmodTypeMask) << kETLmodTypeOffset;
    id_ |= kETLformatV2;
  }
  // v8
  ETLDetId(uint32_t zside,
           uint32_t disc,
           uint32_t discside,
           uint32_t sector,
           uint32_t module,
           uint32_t modtyp,
           uint32_t sensor)
      : MTDDetId(DetId::Forward, ForwardSubdetector::FastTime) {
    id_ |= (MTDType::ETL & kMTDsubdMask) << kMTDsubdOffset | (zside & kZsideMask) << kZsideOffset |
           (encodeSector(disc, discside, sector) & kRodRingMask) << kRodRingOffset |
           (module & kETLmoduleMask) << kETLmoduleOffset | (modtyp & kETLmodTypeMask) << kETLmodTypeOffset |
           (sensor & kETLsensorMask) << kETLsensorOffset;
    id_ |= kETLformatV2;
  }

  // ---------- Common methods ----------

  /** Returns ETL module number. */
  inline int module() const { return (id_ >> kETLmoduleOffset) & kETLmoduleMask; }

  /** Returns ETL module type number. */
  inline int modType() const { return (id_ >> kETLmodTypeOffset) & kETLmodTypeMask; }

  /** Returns ETL module sensor number. */
  inline int sensor() const { return (id_ >> kETLsensorOffset) & kETLsensorMask; }

  ETLDetId geographicalId() const { return id_; }

  // --------- Methods for the TDR ETL model only -----------
  // meaningless for TP model

  // starting from 1
  inline int sector() const { return ((((id_ >> kRodRingOffset) & kRodRingMask) - 1) & kETLsectorMask) + 1; }

  // 0 = front, 1 = back
  inline int discSide() const {
    return ((((id_ >> kRodRingOffset) & kRodRingMask) - 1) >> kETLdiscSideOffset) & kETLdiscSideMask;
  }

  // starting from 1
  inline int nDisc() const {
    return (((((id_ >> kRodRingOffset) & kRodRingMask) - 1) >> kETLnDiscOffset) & kETLnDiscMask) + 1;
  }

  uint32_t newForm(const uint32_t& rawid) {
    uint32_t fixedP = rawid & (0xFFFFFFFF - kETLoldFieldMask);          // unchanged part of id
    uint32_t shiftP = (rawid & kETLoldFieldMask) >> kETLoldToNewShift;  // shifted part
    return ((fixedP | shiftP) | kETLformatV2);
  }
};

std::ostream& operator<<(std::ostream&, const ETLDetId&);

#endif  // DataFormats_ETLDetId_ETLDetId_h
