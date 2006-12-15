#ifndef DataFormats_SiStripCommon_SiStripDetKey_h
#define DataFormats_SiStripCommon_SiStripDetKey_h

#include "DataFormats/SiStripCommon/interface/SiStripEnumeratedTypes.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include <boost/cstdint.hpp>
#include <ostream>

/*
  WHAT ABOUT GOING TO LEVEL OF APV????
  WHAT ABOUT LEVELS ABOVE MODULE??? NEEDS WORK!!!

  can generate another key that is NOT DetId and packs
  sistrip-specific data in a more condensed way, so that all levels
  can be encoded with "all" and "invalid" values, down to level of
  apv. also, need "conversion tool" that re-generates DetId key from
  this new key. this is only way...!!!  maybe can "safeguard" use of
  this key as a DetId by reserving bits 22-24 as a flag (eg, set all
  high), so that if an attempt to build DetId using SiStripDetId
  class, we can understand if key is real DetId or not...

*/

/** */
class SiStripDetKey {
  
 public:
  
  /** Container class holding parameters that identify a logical
      position within the detector to the level of an APV pair. */
  class Path {
  public:
    uint32_t detId_;
    uint16_t apvPair_;
    Path();
    Path( const uint32_t& det_id,
	  const uint16_t& apv_pair );
    bool isEqual( const Path& );
    bool isConsistent( const Path& );
    bool isInvalid() const;
    bool isInvalid( const sistrip::Granularity& ) const;
  };

  // ---------- Returns 32-bit keys based on paths ----------
  
  /** Returns 32-bit key based on Path object. */
  static uint32_t key( const Path& );
  
  /** Returns 32-bit key based on DetId raw value and "APV pair" number. */
  static uint32_t key( const uint32_t& det_id,
		       const uint16_t& apv_pair );

  /** Returns 32-bit key based on DetId object and "APV pair" number. */
  static uint32_t key( const DetId& det_id,
		       const uint16_t& apv_pair );

  // ---------- Returns paths based on 32-bit keys ----------
  
  /** Extracts DetId and "APV pair" number from 32-bit key. */
  static Path path( const uint32_t& det_key );

  /** Extracts DetId and "APV pair" number from DetId object. */
  static Path path( const SiStripDetId& det_id );

  // ---------- Consistency checks between 32-bit keys ----------

  static bool isEqual( const uint32_t& first_key, 
		       const uint32_t& second_key );
  
  static bool isConsistent( const uint32_t& first_key, 
			    const uint32_t& second_key );
  
};

/** Debug info for Path container class. */
std::ostream& operator<< ( std::ostream&, const SiStripDetKey::Path& );

#endif // DataFormats_SiStripCommon_SiStripDetKey_h

