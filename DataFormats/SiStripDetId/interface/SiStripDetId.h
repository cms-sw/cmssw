#ifndef DataFormats_SiStripDetId_SiStripDetId_h
#define DataFormats_SiStripDetId_SiStripDetId_h

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/SiStripEnums.h"
#include <ostream>

class SiStripDetId;

/** Debug info for SiStripDetId class. */
std::ostream &operator<<(std::ostream &, const SiStripDetId &);

/**
    @class SiStripDetId
    @author R.Bainbridge
    @brief Detector identifier class for the strip tracker.
*/
class SiStripDetId : public DetId {
public:
  // ---------- Constructors, enumerated types ----------

  /** Construct a null id */
  SiStripDetId() : DetId() { ; }

  /** Construct from a raw value */
  SiStripDetId(const uint32_t &raw_id) : DetId(raw_id) { ; }

  /** Construct from generic DetId */
  SiStripDetId(const DetId &det_id) : DetId(det_id.rawId()) { ; }

  /** Construct and fill only the det and sub-det fields. */
  SiStripDetId(Detector det, int subdet) : DetId(det, subdet) { ; }

  /** Enumerated type for tracker sub-deteector systems. */
  using SubDetector = SiStripSubdetector::Subdetector;
  static constexpr auto UNKNOWN = SiStripSubdetector::UNKNOWN;
  static constexpr auto TIB = SiStripSubdetector::TIB;
  static constexpr auto TID = SiStripSubdetector::TID;
  static constexpr auto TOB = SiStripSubdetector::TOB;
  static constexpr auto TEC = SiStripSubdetector::TEC;

  // ---------- Common methods ----------

  /** Returns enumerated type specifying sub-detector. */
  inline SubDetector subDetector() const;

  /** Returns enumerated type specifying sub-detector. */
  inline SiStripModuleGeometry moduleGeometry() const;

  /** A non-zero value means a glued module, null means not glued. */
  inline uint32_t glued() const;

  /** A non-zero value means a stereo module, null means not stereo. */
  inline uint32_t stereo() const;

  /** Returns DetId of the partner module if glued, otherwise null. */
  inline uint32_t partnerDetId() const;

  /** Returns strip length of strip tracker sensor, otherwise null. */
  inline double stripLength() const;

  // ---------- Constructors that set "reserved" field ----------

  /** Construct from a raw value and set "reserved" field. */
  SiStripDetId(const uint32_t &raw_id, const uint16_t &reserved) : DetId(raw_id) {
    id_ &= (~static_cast<uint32_t>(reservedMask_ << reservedStartBit_));
    id_ |= ((reserved & reservedMask_) << reservedStartBit_);
  }

  // -----------------------------------------------------------------------------
  //

  /** Construct from generic DetId and set "reserved" field. */
  SiStripDetId(const DetId &det_id, const uint16_t &reserved) : DetId(det_id.rawId()) {
    id_ &= (~static_cast<uint32_t>(reservedMask_ << reservedStartBit_));
    id_ |= ((reserved & reservedMask_) << reservedStartBit_);
  }

  /** Returns value of "reserved" field. */
  inline uint16_t reserved() const;

private:
  /** Position of "reserved" bit field. */
  static const uint16_t reservedStartBit_ = 20;

  /** */
  static const uint32_t sterStartBit_ = 0;

  /** Mask for "reserved" bit field (3-bits wide). */
  static const uint16_t reservedMask_ = 0x7;

  /** */
  static const uint32_t sterMask_ = 0x3;

  static const unsigned layerStartBit_ = 14;
  static const unsigned layerMask_ = 0x7;
  static const unsigned ringStartBitTID_ = 9;
  static const unsigned ringMaskTID_ = 0x3;
  static const unsigned ringStartBitTEC_ = 5;
  static const unsigned ringMaskTEC_ = 0x7;
};

// ---------- inline methods ----------

SiStripDetId::SubDetector SiStripDetId::subDetector() const {
  return static_cast<SiStripDetId::SubDetector>(subdetId());
}

SiStripModuleGeometry SiStripDetId::moduleGeometry() const {
  auto geometry = SiStripModuleGeometry::UNKNOWNGEOMETRY;
  switch (subDetector()) {
    case TIB:
      geometry =
          int((id_ >> layerStartBit_) & layerMask_) < 3 ? SiStripModuleGeometry::IB1 : SiStripModuleGeometry::IB2;
      break;
    case TOB:
      geometry =
          int((id_ >> layerStartBit_) & layerMask_) < 5 ? SiStripModuleGeometry::OB2 : SiStripModuleGeometry::OB1;
      break;
    case TID:
      switch ((id_ >> ringStartBitTID_) & ringMaskTID_) {
        case 1:
          geometry = SiStripModuleGeometry::W1A;
          break;
        case 2:
          geometry = SiStripModuleGeometry::W2A;
          break;
        case 3:
          geometry = SiStripModuleGeometry::W3A;
          break;
      }
      break;
    case TEC:
      switch ((id_ >> ringStartBitTEC_) & ringMaskTEC_) {
        case 1:
          geometry = SiStripModuleGeometry::W1B;
          break;
        case 2:
          geometry = SiStripModuleGeometry::W2B;
          break;
        case 3:
          geometry = SiStripModuleGeometry::W3B;
          break;
        case 4:
          geometry = SiStripModuleGeometry::W4;
          break;
        case 5:
          geometry = SiStripModuleGeometry::W5;
          break;
        case 6:
          geometry = SiStripModuleGeometry::W6;
          break;
        case 7:
          geometry = SiStripModuleGeometry::W7;
          break;
      }
    case UNKNOWN:
    default:;
  }
  return geometry;
}

uint32_t SiStripDetId::glued() const {
  uint32_t testId = (id_ >> sterStartBit_) & sterMask_;
  return (testId == 0) ? 0 : (id_ - testId);
}

uint32_t SiStripDetId::stereo() const { return (((id_ >> sterStartBit_) & sterMask_) == 1) ? 1 : 0; }

uint32_t SiStripDetId::partnerDetId() const {
  uint32_t testId = (id_ >> sterStartBit_) & sterMask_;
  if (testId == 1) {
    testId = id_ + 1;
  } else if (testId == 2) {
    testId = id_ - 1;
  } else {
    testId = 0;
  }
  return testId;
}

double SiStripDetId::stripLength() const { return 0.; }

uint16_t SiStripDetId::reserved() const { return static_cast<uint16_t>((id_ >> reservedStartBit_) & reservedMask_); }

#endif  // DataFormats_SiStripDetId_SiStripDetId_h
