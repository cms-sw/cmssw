#ifndef DataFormats_SiStripCommon_SiStripFecKey_H
#define DataFormats_SiStripCommon_SiStripFecKey_H

#include "DataFormats/SiStripCommon/interface/SiStripEnumeratedTypes.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <ostream>

class SiStripFecKey {
  
 public:
  
  /** Container class holding parameters that identify a logical
      position within the control structure to the level of an APV. */
  class Path { 
  public:
    uint16_t fecCrate_; // FEC crate [1-4]
    uint16_t fecSlot_;  // FEC slot [1-21]
    uint16_t fecRing_;  // FEC ring [1-8]
    uint16_t ccuAddr_;  // CCU module [1-127]
    uint16_t ccuChan_;  // FE module [16-31]
    uint16_t channel_;  // LLD/APV [1-3,32-37]
    Path();
    Path( const uint16_t& fec_crate, 
	  const uint16_t& fec_slot, 
	  const uint16_t& fec_ring, 
	  const uint16_t& ccu_addr, 
	  const uint16_t& ccu_chan,
	  const uint16_t& channel = uint16_t(sistrip::invalid_) );
    bool isEqual( const Path& ) const;
    bool isConsistent( const Path& ) const;
    bool isInvalid() const;
    bool isInvalid( const sistrip::Granularity& ) const;
  };
  
  // ---------- Returns 32-bit keys based on paths ----------
  
  /** Returns 32-bit key based on Path object. */
  static uint32_t key( const Path& );
  
  /** Returns 32-bit key based on control logical structure. */
  static uint32_t key( uint16_t fec_crate = sistrip::invalid_, 
		       uint16_t fec_slot  = sistrip::invalid_, 
		       uint16_t fec_ring  = sistrip::invalid_, 
		       uint16_t ccu_addr  = sistrip::invalid_, 
		       uint16_t ccu_chan  = sistrip::invalid_,
		       uint16_t channel   = sistrip::invalid_ );

  // ---------- Returns paths based on 32-bit keys ----------

  /** Extracts control logical structure from 32-bit key. */
  static Path path( uint32_t fec_key );
  
  // ---------- Consistency checks between 32-bit keys ----------

  static bool isEqual( const uint32_t& first_key, 
		       const uint32_t& second_key );
  
  static bool isConsistent( const uint32_t& first_key, 
			    const uint32_t& second_key );
  
 public:

  static const uint16_t fecCrateOffset_ = 29;
  static const uint16_t fecSlotOffset_  = 24;
  static const uint16_t fecRingOffset_  = 20;
  static const uint16_t ccuAddrOffset_  = 12;
  static const uint16_t ccuChanOffset_  =  7;
  static const uint16_t channelOffset_  =  0;
  
  static const uint16_t fecCrateMask_ = 0x07;
  static const uint16_t fecSlotMask_  = 0x1F;
  static const uint16_t fecRingMask_  = 0x0F;
  static const uint16_t ccuAddrMask_  = 0xFF;
  static const uint16_t ccuChanMask_  = 0x1F;
  static const uint16_t channelMask_  = 0x7F;
  
};

/** Debug info for Path container class. */
std::ostream& operator<< ( std::ostream&, const SiStripFecKey::Path& );

#endif // DataFormats_SiStripCommon_SiStripFecKey_H
