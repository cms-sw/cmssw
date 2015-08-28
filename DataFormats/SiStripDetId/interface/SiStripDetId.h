#ifndef DataFormats_SiStripDetId_SiStripDetId_h
#define DataFormats_SiStripDetId_SiStripDetId_h

#include "DataFormats/DetId/interface/DetId.h"
#include <ostream>

class SiStripDetId;

/** Debug info for SiStripDetId class. */
std::ostream& operator<< ( std::ostream&, const SiStripDetId& );

/** 
    @class SiStripDetId
    @author R.Bainbridge
    @brief Detector identifier class for the strip tracker.
*/
class SiStripDetId : public DetId {
  
 public:

  // ---------- Constructors, enumerated types ----------
  
  /** Construct a null id */
  SiStripDetId()  : DetId() {;}

  /** Construct from a raw value */
  SiStripDetId( const uint32_t& raw_id ) : DetId( raw_id ) {;}

  /** Construct from generic DetId */
  SiStripDetId( const DetId& det_id )  : DetId( det_id.rawId() ) {;}

  /** Construct and fill only the det and sub-det fields. */
  SiStripDetId( Detector det, int subdet ) : DetId( det, subdet ) {;}

  /** Enumerated type for tracker sub-deteector systems. */
  enum SubDetector { UNKNOWN=0, TIB=3, TID=4, TOB=5, TEC=6 };
  
  /** Enumerated type for tracker module geometries. */
  enum ModuleGeometry {UNKNOWNGEOMETRY, IB1, IB2, OB1, OB2, W1A, W2A, W3A, W1B, W2B, W3B, W4, W5, W6, W7};

  // ---------- Common methods ----------

  /** Returns enumerated type specifying sub-detector. */
  inline SubDetector subDetector() const;
  
  /** Returns enumerated type specifying sub-detector. */
  inline ModuleGeometry moduleGeometry() const;

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
  SiStripDetId( const uint32_t& raw_id, 
		const uint16_t& reserved )
    : DetId( raw_id ) 
  {
    id_ &= ( ~static_cast<uint32_t>(reservedMask_<<reservedStartBit_) );
    id_ |= ( ( reserved & reservedMask_ ) << reservedStartBit_ );
  }
  
  // -----------------------------------------------------------------------------
  //
  
  /** Construct from generic DetId and set "reserved" field. */
  SiStripDetId( const DetId& det_id, 
		const uint16_t& reserved )
    : DetId( det_id.rawId() ) 
  {
    id_ &= ( ~static_cast<uint32_t>(reservedMask_<<reservedStartBit_) );
    id_ |= ( ( reserved & reservedMask_ ) << reservedStartBit_ );
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
  static const unsigned ringStartBitTID_= 9;
  static const unsigned ringMaskTID_= 0x3;
  static const unsigned ringStartBitTEC_= 5;
  static const unsigned ringMaskTEC_= 0x7;
  
};

// ---------- inline methods ----------

SiStripDetId::SubDetector SiStripDetId::subDetector() const {
  SiStripDetId::SubDetector result;
  if ( det() == DetId::Tracker) {
    if ( subdetId() == static_cast<int>(SiStripDetId::TEC) ) {
      result = SiStripDetId::TEC;
    } else if ( subdetId() == static_cast<int>(SiStripDetId::TID) ) {
      result = SiStripDetId::TID;
    } else if ( subdetId() == static_cast<int>(SiStripDetId::TOB) ) {
      result = SiStripDetId::TOB;
    } else if ( subdetId() == static_cast<int>(SiStripDetId::TIB) ) {
      result = SiStripDetId::TIB;
    } else {
      result = SiStripDetId::UNKNOWN;
    }
  } else {
    result = SiStripDetId::UNKNOWN;
  }
  return result;
}

SiStripDetId::ModuleGeometry SiStripDetId::moduleGeometry() const {
  SiStripDetId::ModuleGeometry geometry = UNKNOWNGEOMETRY;
  switch(subDetector()) {
  case TIB: geometry = int((id_>>layerStartBit_) & layerMask_)<3? IB1 : IB2;
    break;
  case TOB: geometry = int((id_>>layerStartBit_) & layerMask_)<5? OB2 : OB1;
    break;
  case TID: switch ((id_>>ringStartBitTID_) & ringMaskTID_) {
    case 1: geometry = W1A;
      break;
    case 2: geometry = W2A;
      break;
    case 3: geometry = W3A;
      break;
    }
    break;
  case TEC: switch ((id_>>ringStartBitTEC_) & ringMaskTEC_) {
    case 1: geometry = W1B;
      break;
    case 2: geometry = W2B;
      break;
    case 3: geometry = W3B;
      break;
    case 4: geometry = W4;
      break;
    case 5: geometry = W5;
      break;
    case 6: geometry = W6;
      break;
    case 7: geometry = W7;
      break;
    }
  case UNKNOWN: default:;  
  }
  return geometry;
}

uint32_t SiStripDetId::glued() const {
  uint32_t testId = (id_>>sterStartBit_) & sterMask_;
  return ( testId == 0 ) ? 0 : (id_ - testId);
}
 
uint32_t SiStripDetId::stereo() const {
  return ( ((id_>>sterStartBit_) & sterMask_) == 1 ) ? 1 : 0;
}
 
uint32_t SiStripDetId::partnerDetId() const {
  uint32_t testId = (id_>>sterStartBit_) & sterMask_;
  if ( testId == 1 ) {
    testId = id_ + 1;
  } else if ( testId == 2 ) {
    testId = id_ - 1;
  } else { testId = 0; }
  return testId;
}
 
double SiStripDetId::stripLength() const {
  return 0.;
}


uint16_t SiStripDetId::reserved() const { 
  return static_cast<uint16_t>( (id_>>reservedStartBit_) & reservedMask_ );
}

#endif // DataFormats_SiStripDetId_SiStripDetId_h

