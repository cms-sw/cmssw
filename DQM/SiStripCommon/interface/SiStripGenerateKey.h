#ifndef DQM_SiStripCommon_SiStripGenerateKey_H
#define DQM_SiStripCommon_SiStripGenerateKey_H

#include <boost/cstdint.hpp>

using namespace std;

class SiStripGenerateKey {
  
 public:

  /** Returns reserved value (0xFFFF) to represent "NOT SPECIFIED". */
  inline static const uint16_t& all() { return all_; }
 
  /** Returns a 32-bit key that uniquely identifies a FED channel,
      built from a FED id and channel number. */
  static uint32_t fed( uint32_t fed_id, uint32_t fed_ch );
  
  /** Returns a pair of FED id and channel, extracted from a 32-bit
      key that uniquely identifies a FED channel. */
  static pair<uint32_t,uint32_t> fed( uint32_t fed_key );
  
  /** 32-bit key that uniquely identifies a module. The key is built
      from the crate, FEC, ring, CCU and module addresses. */
  static uint32_t module( uint16_t fec_crate = all_, 
			  uint16_t fec_slot = all_, 
			  uint16_t fec_ring = all_, 
			  uint16_t ccu_addr = all_, 
			  uint16_t ccu_chan = all_ );
  
 private: 
  
  /** Reserved value (0xFFFF) to represent "NOT SPECIFIED". */
  static const uint16_t all_;
  
};

#endif // DQM_SiStripCommon_SiStripGenerateKey_H


