#ifndef DataFormats_SiStripDetId_SiStripReadoutKey_H
#define DataFormats_SiStripDetId_SiStripReadoutKey_H

#include <boost/cstdint.hpp>

using namespace std;

class SiStripReadoutKey {
  
 public:
  
  /** Reserved value (0xFFFF) to represent "NOT SPECIFIED". */
  static const uint16_t all_;
  
  /** Simple struct to hold parameters that uniquely identify a
      hardware component within the readout system, down to the level
      of a FED channel. */
  struct ReadoutPath { 
    uint16_t fedId_;
    uint16_t fedCh_; 
  };
  
  /** Returns a 32-bit key that uniquely identifies a FED channel,
      built from a FED id and channel number. */
  static uint32_t key( uint16_t fed_id = all_, 
		       uint16_t fed_channel = all_ );
  
  /** Returns the FED id and channel that are extracted from a 32-bit
      key and uniquely identify a hardware component of the readout
      system, down to the level of a FED channel. */
  static ReadoutPath path( uint32_t key );
  
};

#endif // DataFormats_SiStripDetId_SiStripReadoutKey_H


