#include "DQM/SiStripCommon/interface/SiStripGenerateKey.h"
#include <iostream>
#include <sstream>

// -----------------------------------------------------------------------------
// definition of static (public) data
const uint16_t SiStripGenerateKey::all_ = 0xFFFF;

// -----------------------------------------------------------------------------
//
uint32_t SiStripGenerateKey::module( uint16_t fec_crate,
				     uint16_t fec_slot,
				     uint16_t fec_ring,
				     uint16_t ccu_addr,
				     uint16_t ccu_chan ) {
  return ( static_cast<uint32_t>( (fec_crate & 0x0F) << 28 ) | 
	   static_cast<uint32_t>( (fec_slot & 0xFF)  << 20 ) | 
	   static_cast<uint32_t>( (fec_ring & 0x0F)  << 16 ) | 
	   static_cast<uint32_t>( (ccu_addr & 0xFF)  <<  8 ) | 
	   static_cast<uint32_t>( (ccu_chan & 0xFF)  <<  0 ) );
}

// -----------------------------------------------------------------------------
//
uint32_t SiStripGenerateKey::fed( uint32_t fed_id, 
				  uint32_t fed_ch ) {
  return ( ( fed_id & 0xFFFF ) << 16 ) | ( ( fed_ch & 0xFFFF ) << 0 );
}

// -----------------------------------------------------------------------------
//
pair<uint32_t,uint32_t> SiStripGenerateKey::fed( uint32_t fed_key ) {
  return pair<uint32_t,uint32_t>( ( fed_key >> 16 ) & 0xFFFF, fed_key & 0xFFFF );
}

