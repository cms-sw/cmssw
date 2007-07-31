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
  SiStripDetId();

  /** Construct from a raw value */
  SiStripDetId( const uint32_t& raw_id );

  /** Construct from generic DetId */
  SiStripDetId( const DetId& );

  /** Construct and fill only the det and sub-det fields. */
  SiStripDetId( Detector det, int subdet );

  /** Enumerated type for tracker sub-deteector systems. */
  enum SubDetector { UNKNOWN=0, TIB=3, TID=4, TOB=5, TEC=6 };
  
  // ---------- Common methods ----------

  /** Returns enumerated type specifying sub-detector. */
  inline SubDetector subDetector() const;
  
  /** A non-zero value means a glued module, null means not glued. */
  inline virtual uint32_t glued() const;
  
  /** A non-zero value means a stereo module, null means not stereo. */
  inline virtual uint32_t stereo() const;

  /** Returns DetId of the partner module if glued, otherwise null. */
  inline virtual uint32_t partnerDetId() const;
 
  /** Returns strip length of strip tracker sensor, otherwise null. */
  inline virtual double stripLength() const;

  // ---------- Constructors that set "reserved" field ----------
  
  /** Construct from a raw value and set "reserved" field. */
  SiStripDetId( const uint32_t& raw_id, 
		const uint16_t& reserved );
  
  /** Construct from generic DetId and set "reserved" field. */
  SiStripDetId( const DetId& det_id, 
		const uint16_t& reserved );
  
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

