#ifndef DataFormats_ETLDetId_ETLDetId_h
#define DataFormats_ETLDetId_ETLDetId_h

#include "DataFormats/ForwardDetId/interface/MTDDetId.h"
#include <ostream>

/** 
    @class ETLDetId
    @brief Detector identifier class for the Endcap Timing Layer.

    bit 15-7 : module sequential number
    bit 6-5  : module type (unused so far)
*/

class ETLDetId : public MTDDetId {
public:
  static const uint32_t kETLmoduleOffset = 7;
  static const uint32_t kETLmoduleMask = 0x1FF;
  static const uint32_t kETLmodTypeOffset = 5;
  static const uint32_t kETLmodTypeMask = 0x3;

  static constexpr int kETLv1maxRing = 11;
  static constexpr int kETLv1maxModule = 176;

  /// constants for the TDR ETL model
  static const uint32_t kETLnDiscOffset = 3;
  static const uint32_t kETLnDiscMask = 0x1;
  static const uint32_t kETLdiscSideOffset = 2;
  static const uint32_t kETLdiscSideMask = 0x1;
  static const uint32_t kETLquarterMask = 0x3;

  static constexpr int kETLv2maxRing = 16;
  static constexpr int kETLv2maxModule = 292;

  // ---------- Constructors, enumerated types ----------

  /** Construct a null id */
  ETLDetId() : MTDDetId(DetId::Forward, ForwardSubdetector::FastTime) {
    id_ |= (MTDType::ETL & kMTDsubdMask) << kMTDsubdOffset;
  }

  /** Construct from a raw value */
  ETLDetId(const uint32_t& raw_id) : MTDDetId(raw_id) { ; }

  /** Construct from generic DetId */
  ETLDetId(const DetId& det_id) : MTDDetId(det_id.rawId()) { ; }

  /** Construct and fill only the det and sub-det fields. */
  ETLDetId(uint32_t zside, uint32_t ring, uint32_t module, uint32_t modtyp)
      : MTDDetId(DetId::Forward, ForwardSubdetector::FastTime) {
    id_ |= (MTDType::ETL & kMTDsubdMask) << kMTDsubdOffset | (zside & kZsideMask) << kZsideOffset |
           (ring & kRodRingMask) << kRodRingOffset | (module & kETLmoduleMask) << kETLmoduleOffset |
           (modtyp & kETLmodTypeMask) << kETLmodTypeOffset;
  }

  // ---------- Common methods ----------

  /** Returns ETL module number. */
  inline int module() const { return (id_ >> kETLmoduleOffset) & kETLmoduleMask; }

  /** Returns ETL module type number. */
  inline int modType() const { return (id_ >> kETLmodTypeOffset) & kETLmodTypeMask; }

  ETLDetId geographicalId() const;

  // --------- Methods for the TDR ETL model only -----------
  // meaningless for TP model

  // starting from 1
  inline int quarter() const { return ((((id_ >> kRodRingOffset) & kRodRingMask) - 1) & kETLquarterMask) + 1; }

  // 0 = front, 1 = back
  inline int discSide() const {
    return ((((id_ >> kRodRingOffset) & kRodRingMask) - 1) >> kETLdiscSideOffset) & kETLdiscSideMask;
  }

  // starting from 1
  inline int nDisc() const {
    return (((((id_ >> kRodRingOffset) & kRodRingMask) - 1) >> kETLnDiscOffset) & kETLnDiscMask) + 1;
  }
};

std::ostream& operator<<(std::ostream&, const ETLDetId&);

#endif  // DataFormats_ETLDetId_ETLDetId_h
