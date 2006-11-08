#ifndef DataFormats_SiStripCommon_SiStripDetKey_h
#define DataFormats_SiStripCommon_SiStripDetKey_h

#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include <boost/cstdint.hpp>
#include <ostream>

/** */
class SiStripDetKey {
  
 public:

  /** Simple container class holding DetId and "APV pair" number. */
  class Path {
  public:
    uint32_t detId_;
    uint16_t apvPair_;
    Path() : detId_(0), apvPair_(0) {;}
    Path( const uint32_t& det_id,
	  const uint16_t& apv_pair ) 
      : detId_(det_id), apvPair_(apv_pair) {;}
  };

  /** Returns 32-bit key based on DetId and "APV pair" number. */
  static uint32_t key( const uint32_t& det_id,
		       const uint16_t& apv_pair );
  /** Returns 32-bit key based on DetId and "APV pair" number. */
  static uint32_t key( const DetId& det_id,
		       const uint16_t& apv_pair );
  
  /** Returns container class holding DetId and "APV pair" number. */
  static Path path( const uint32_t& det_id );
  /** Returns container class holding DetId and "APV pair" number. */
  static Path path( const SiStripDetId& det_id );

 private: 

  /*   static const uint16_t offset_ = 20; */
  /*   static const uint16_t mask_   = 0x7; */
  
};

/** Debug info for Path container class. */
std::ostream& operator<< ( std::ostream&, const SiStripDetKey::Path& );

#endif // DataFormats_SiStripCommon_SiStripDetKey_h

