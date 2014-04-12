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
  if( det() == DetId::Tracker &&
      subdetId() == static_cast<int>(SiStripDetId::TIB) ) {
    return SiStripDetId::TIB;
  } else if ( det() == DetId::Tracker &&
	      subdetId() == static_cast<int>(SiStripDetId::TID) ) {
    return SiStripDetId::TID;
  } else if ( det() == DetId::Tracker &&
	      subdetId() == static_cast<int>(SiStripDetId::TOB) ) {
    return SiStripDetId::TOB;
  } else if ( det() == DetId::Tracker &&
	      subdetId() == static_cast<int>(SiStripDetId::TEC) ) {
    return SiStripDetId::TEC;
  } else {
    return SiStripDetId::UNKNOWN;
  }
}

SiStripDetId::ModuleGeometry SiStripDetId::moduleGeometry() const {
  switch(subDetector()) {
  case TIB: return int((id_>>layerStartBit_) & layerMask_)<3? IB1 : IB2;
  case TOB: return int((id_>>layerStartBit_) & layerMask_)<5? OB2 : OB1;
  case TID: switch ((id_>>ringStartBitTID_) & ringMaskTID_) {
    case 1: return W1A;
    case 2: return W2A;
    case 3: return W3A;
    }
  case TEC: switch ((id_>>ringStartBitTEC_) & ringMaskTEC_) {
    case 1: return W1B;
    case 2: return W2B;
    case 3: return W3B;
    case 4: return W4;
    case 5: return W5;
    case 6: return W6;
    case 7: return W7;
    }
  case UNKNOWN: default: return UNKNOWNGEOMETRY;
  }
}

uint32_t SiStripDetId::glued() const {
  if ( ((id_>>sterStartBit_) & sterMask_ ) == 1 ) {
    return ( id_ - 1 );
  } else if ( ((id_>>sterStartBit_) & sterMask_ ) == 2 ) {
    return ( id_ - 2 );
  } else { return 0; }
}
 
uint32_t SiStripDetId::stereo() const {
  if ( ((id_>>sterStartBit_ ) & sterMask_ ) == 1 ) {
    return ( (id_>>sterStartBit_) & sterMask_ );
  } else { return 0; }
}
 
uint32_t SiStripDetId::partnerDetId() const {
  if ( ((id_>>sterStartBit_) & sterMask_ ) == 1 ) {
    return ( id_ + 1 );
  } else if ( ((id_>>sterStartBit_) & sterMask_ ) == 2 ) {
    return ( id_ - 1 );
  } else { return 0; }
}
 
double SiStripDetId::stripLength() const {
  return 0.;
}


uint16_t SiStripDetId::reserved() const { 
  return static_cast<uint16_t>( (id_>>reservedStartBit_) & reservedMask_ );
}

#endif // DataFormats_SiStripDetId_SiStripDetId_h

