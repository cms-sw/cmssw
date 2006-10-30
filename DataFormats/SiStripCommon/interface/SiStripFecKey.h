#ifndef DataFormats_SiStripCommon_SiStripFecKey_H
#define DataFormats_SiStripCommon_SiStripFecKey_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <ostream>

class SiStripFecKey {
  
 public:
  
  /** Simple container class that holds parameters that uniquely
      identify an APV or LLD channel within the control system. */
  class Path { 
  public:
    uint16_t fecCrate_; // FEC crate [1-4]: 3-bits, 0=all, 0x07=invalid
    uint16_t fecSlot_;  // FEC slot [1-21]: 5-bits, 0=all, 0x1F=invalid
    uint16_t fecRing_;  // FEC ring [1-8]: 4-bits, 0=all, 0x0F=invalid
    uint16_t ccuAddr_;  // CCU module [1-127]: 8-bits, 0=all, 0xFF=invalid
    uint16_t ccuChan_;  // FE module [16-31]: 5-bits, 0=all, 0x1F=invalid
    uint16_t channel_;  // LLD/APV [1-3,32-37]: 7-bits, 0=all, 0x7F=invalid
    Path() : 
      fecCrate_(sistrip::invalid_), fecSlot_(sistrip::invalid_),
      fecRing_(sistrip::invalid_), ccuAddr_(sistrip::invalid_),
      ccuChan_(sistrip::invalid_), channel_(sistrip::invalid_) {;}
    Path( const uint16_t& fec_crate, 
	  const uint16_t& fec_slot, 
	  const uint16_t& fec_ring, 
	  const uint16_t& ccu_addr, 
	  const uint16_t& ccu_chan,
	  uint16_t channel = sistrip::invalid_ ) :
      fecCrate_(fec_crate), fecSlot_(fec_slot),
      fecRing_(fec_ring), ccuAddr_(ccu_addr),
      ccuChan_(ccu_chan), channel_(channel) {;}
  };
  
  /** Returns the parameters that uniquely identify an APV or LLD
      channel within the control system. */
  static Path path( uint32_t key );

  /** Creates a 32-bit key that uniquely identifies an APV or LLD
      channel within the strip tracker control system. */
  static uint32_t key( const Path& );
  
  /** Creates a 32-bit key that uniquely identifies an APV or LLD
      channel within the strip tracker control system. */
  static uint32_t key( uint16_t fec_crate = sistrip::invalid_, 
		       uint16_t fec_slot  = sistrip::invalid_, 
		       uint16_t fec_ring  = sistrip::invalid_, 
		       uint16_t ccu_addr  = sistrip::invalid_, 
		       uint16_t ccu_chan  = sistrip::invalid_,
		       uint16_t channel   = sistrip::invalid_ );
  
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
