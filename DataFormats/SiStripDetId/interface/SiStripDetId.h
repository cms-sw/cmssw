#ifndef DataFormats_SiStripDetId_SiStripDetId_h
#define DataFormats_SiStripDetId_SiStripDetId_h

#include "DataFormats/DetId/interface/DetId.h"
#include <boost/cstdint.hpp>
#include <ostream>

/** Det identifier class for the strip tracker */
class SiStripDetId : public DetId {
  
 public:
  
  /** Construct a null id */
  SiStripDetId();
  /** Construct from a raw value */
  SiStripDetId( const uint32_t& raw_id );
  /** Construct from generic DetId */
  SiStripDetId( const DetId& );
  /** Construct and fill only the det and sub-det fields. */
  SiStripDetId( Detector det, int subdet );
  
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
  /** Mask for "reserved" bit field (3-bits wide). */ 
  static const uint16_t reservedMask_ = 0x7;
  
};

uint16_t SiStripDetId::reserved() const { 
  return static_cast<uint16_t>( (id_>>reservedStartBit_) & reservedMask_ );
}

/** Debug info for SiStripDetId class. */
std::ostream& operator<< ( std::ostream&, const SiStripDetId& );

#endif // DataFormats_SiStripDetId_SiStripDetId_h

