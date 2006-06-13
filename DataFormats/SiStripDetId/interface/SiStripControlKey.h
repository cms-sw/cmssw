#ifndef DataFormats_SiStripDetId_SiStripControlKey_H
#define DataFormats_SiStripDetId_SiStripControlKey_H

#include <boost/cstdint.hpp>

using namespace std;

class SiStripControlKey {
  
 public:

  /** Reserved value (0xFFFF) to represent "NOT SPECIFIED". */
  static const uint16_t all_;
  
  /** Simple struct to hold parameters that uniquely identify a
      hardware component within the control system, down to the level
      of a front-end device or channel. */
  struct ControlPath { 
    uint16_t fecCrate_; // [0-3]   //@@ crate numbering from zero!
    uint16_t fecSlot_;  // [0-31] 
    uint16_t fecRing_;  // [0-15]
    uint16_t ccuAddr_;  // [0-127]
    uint16_t ccuChan_;  // [0-255]
    uint16_t channel_;  // [0-63]  //@@ I2C addresses only up to 63!
  };
  
  /** 32-bit key that uniquely identifies a hardware component of the
      control system, down to the level of an LLD channel within a
      module. The key is built from the FEC crate, slot and ring, CCU
      address and channel, and front-end device or channel. */
  static uint32_t key( uint16_t fec_crate = all_, 
		       uint16_t fec_slot  = all_, 
		       uint16_t fec_ring  = all_, 
		       uint16_t ccu_addr  = all_, 
		       uint16_t ccu_chan  = all_,
		       uint16_t channel   = all_ );
  
  /** Returns the FEC crate, slot and ring, CCU address and channel,
      and LLD channel that are extracted from a 32-bit key and
      uniquely identify a hardware component of the control system,
      down to the level of a front-end device or channel. */
  static const ControlPath& path( uint32_t key );
  
};

#endif // DataFormats_SiStripDetId_SiStripControlKey_H


