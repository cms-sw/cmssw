#include "DataFormats/SiStripDetId/interface/SiStripControlKey.h"
#include <iostream>
#include <sstream>

// -----------------------------------------------------------------------------
// definition of static (public) data
const uint16_t SiStripControlKey::all_ = 0xFFFF;

// -----------------------------------------------------------------------------
//
uint32_t SiStripControlKey::key( uint16_t fec_crate,
				 uint16_t fec_slot,
				 uint16_t fec_ring,
				 uint16_t ccu_addr,
				 uint16_t ccu_chan,
				 uint16_t lld_chan ) {
  return( static_cast<uint32_t>( (fec_crate & 0x03) << 30 ) | 
	  static_cast<uint32_t>( (fec_slot  & 0xFF) << 22 ) | 
	  static_cast<uint32_t>( (fec_ring  & 0x0F) << 18 ) | 
	  static_cast<uint32_t>( (ccu_addr  & 0xFF) << 10 ) | 
	  static_cast<uint32_t>( (ccu_chan  & 0xFF) <<  2 ) | 
	  static_cast<uint32_t>( (lld_chan  & 0x03) <<  0 ) );
}

// -----------------------------------------------------------------------------
//
SiStripControlKey::ControlPath SiStripControlKey::path( uint32_t key ) {
  ControlPath path;
  path.fecCrate_ = (key>>30) & 0x03;
  path.fecSlot_  = (key>>22) & 0xFF;
  path.fecRing_  = (key>>18) & 0x0F;
  path.ccuAddr_  = (key>>10) & 0xFF;
  path.ccuChan_  = (key>> 2) & 0xFF;
  path.lldChan_  = (key>> 0) & 0x03;
  if ( path.fecCrate_ == 0x03 ) { path.fecCrate_ = all_; } 
  if ( path.fecSlot_  == 0xFF ) { path.fecSlot_  = all_; } 
  if ( path.fecRing_  == 0x0F ) { path.fecRing_  = all_; } 
  if ( path.ccuAddr_  == 0xFF ) { path.ccuAddr_  = all_; } 
  if ( path.ccuChan_  == 0xFF ) { path.ccuChan_  = all_; } 
  if ( path.lldChan_  == 0x03 ) { path.lldChan_  = all_; } 
  return path;  
}
