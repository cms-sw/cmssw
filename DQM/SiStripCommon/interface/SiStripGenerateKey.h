#ifndef DQM_SiStripCommon_SiStripGenerateKey_H
#define DQM_SiStripCommon_SiStripGenerateKey_H

#include <boost/cstdint.hpp>

using namespace std;

class SiStripGenerateKey {
  
 public:

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

  /** Returns reserved value (0xFFFF) to represent "NOT SPECIFIED". */
  inline static const uint16_t& all() { return all_; }
 
  /** Returns a 32-bit key that uniquely identifies a FED channel,
      built from a FED id and channel number. */
  static uint32_t fedKey( uint16_t fed_id = all_, 
			  uint16_t fed_channel = all_ );
  
  /** Returns a pair of FED id and channel, extracted from a 32-bit
      key, that uniquely identify a FED channel. */
  static pair<uint16_t,uint16_t> fedChannel( uint32_t fed_key );
  
  /** 32-bit key that uniquely identifies a hardware component of the
      control system, down to the level of an LLD channel within a
      module. The key is built from the FEC crate, slot and ring, CCU
      address and channel, and the LLD channel. */
  static uint32_t controlKey( uint16_t fec_crate = all_, 
			      uint16_t fec_slot  = all_, 
			      uint16_t fec_ring  = all_, 
			      uint16_t ccu_addr  = all_, 
			      uint16_t ccu_chan  = all_,
			      uint16_t lld_chan  = all_ );
  
  /** Returns the FEC crate, slot and ring, CCU address and channel,
      and LLD channel that are extracted from a 32-bit key and
      uniquely identify a hardware component of the control system,
      down to the level of an LLD channel. */
  static ControlPath controlPath( uint32_t key );
  
 private: 
  
  /** Reserved value (0xFFFF) to represent "NOT SPECIFIED". */
  static const uint16_t all_;
  
};

#endif // DQM_SiStripCommon_SiStripGenerateKey_H


