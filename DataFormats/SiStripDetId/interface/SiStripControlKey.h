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
      of an LLD channel. */
  struct ControlPath { 
    uint16_t fecCrate_;
    uint16_t fecSlot_; 
    uint16_t fecRing_; 
    uint16_t ccuAddr_;
    uint16_t ccuChan_;
    uint16_t lldChan_;
  };

  /** 32-bit key that uniquely identifies a hardware component of the
      control system, down to the level of an LLD channel within a
      module. The key is built from the FEC crate, slot and ring, CCU
      address and channel, and the LLD channel. */
  static uint32_t key( uint16_t fec_crate = all_, 
		       uint16_t fec_slot  = all_, 
		       uint16_t fec_ring  = all_, 
		       uint16_t ccu_addr  = all_, 
		       uint16_t ccu_chan  = all_,
		       uint16_t lld_chan  = all_ );
  
  /** Returns the FEC crate, slot and ring, CCU address and channel,
      and LLD channel that are extracted from a 32-bit key and
      uniquely identify a hardware component of the control system,
      down to the level of an LLD channel. */
  static ControlPath path( uint32_t key );
  
};

#endif // DataFormats_SiStripDetId_SiStripControlKey_H


