#ifndef DataFormats_SiStripDetId_SiStripFecKey_H
#define DataFormats_SiStripDetId_SiStripFecKey_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <ostream>

class SiStripFecKey {
  
 public:
  
  /** Simple container class that holds parameters that uniquely
      identify an APV or LLD channel within the control system. */
  class Path { 
  public:
    uint16_t fecCrate_; // [0-3]   //@@ crate numbering from zero!
    uint16_t fecSlot_;  // [0-31] 
    uint16_t fecRing_;  // [0-15]
    uint16_t ccuAddr_;  // [0-127]
    uint16_t ccuChan_;  // [0-255]
    uint16_t channel_;  // [0-63]  //@@ I2C addresses only up to 63!
    Path() : 
      fecCrate_(sistrip::invalid_), fecSlot_(sistrip::invalid_),
      fecRing_(sistrip::invalid_), ccuAddr_(sistrip::invalid_),
      ccuChan_(sistrip::invalid_), channel_(sistrip::invalid_) {;}
    Path( uint16_t fec_crate, 
	  uint16_t fec_slot, 
	  uint16_t fec_ring, 
	  uint16_t ccu_addr, 
	  uint16_t ccu_chan,
	  uint16_t channel = sistrip::invalid_ ) :
      fecCrate_(fec_crate), fecSlot_(fec_slot),
      fecRing_(fec_ring), ccuAddr_(ccu_addr),
      ccuChan_(ccu_chan), channel_(channel) {;}
  };
  
  /** Creates a 32-bit key that uniquely identifies an APV or LLD
      channel within the strip tracker control system. */
  static uint32_t key( uint16_t fec_crate = sistrip::invalid_, 
		       uint16_t fec_slot  = sistrip::invalid_, 
		       uint16_t fec_ring  = sistrip::invalid_, 
		       uint16_t ccu_addr  = sistrip::invalid_, 
		       uint16_t ccu_chan  = sistrip::invalid_,
		       uint16_t channel   = sistrip::invalid_ );
  
  /** Creates a 32-bit key that uniquely identifies an APV or LLD
      channel within the strip tracker control system. */
  static uint32_t key( const Path& );
  
  /** Returns the parameters that uniquely identify an APV or LLD
      channel within the control system. */
  static Path path( uint32_t key );
  
 private:

  static const uint16_t fecCrateOffset_ = 30;
  static const uint16_t fecSlotOffset_  = 25;
  static const uint16_t fecRingOffset_  = 21;
  static const uint16_t ccuAddrOffset_  = 14;
  static const uint16_t ccuChanOffset_  =  6;
  static const uint16_t channelOffset_  =  0;

  static const uint16_t fecCrateMask_ = 0x03;
  static const uint16_t fecSlotMask_  = 0x1F;
  static const uint16_t fecRingMask_  = 0x0F;
  static const uint16_t ccuAddrMask_  = 0x7F;
  static const uint16_t ccuChanMask_  = 0xFF;
  static const uint16_t channelMask_  = 0x3F;
  
};

/** Debug info for Path container class. */
std::ostream& operator<< ( std::ostream&, const SiStripFecKey::Path& );

#endif // DataFormats_SiStripDetId_SiStripFecKey_H


