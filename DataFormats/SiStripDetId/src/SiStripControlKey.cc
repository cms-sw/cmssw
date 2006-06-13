#include "DataFormats/SiStripDetId/interface/SiStripControlKey.h"
#include <iostream>
#include <sstream>

//@@ ALTERNATIVE METHOD?...  
// can use 8 bits for device address, start crate/slot/ring numbering
// from zero and use only slot numbers 0-15 in crate, to gain 3 bits

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
				 uint16_t channel ) {
  return( static_cast<uint32_t>( (fec_crate & 0x03) << 30 ) | 
	  static_cast<uint32_t>( (fec_slot  & 0x1F) << 25 ) | 
	  static_cast<uint32_t>( (fec_ring  & 0x0F) << 21 ) | 
	  static_cast<uint32_t>( (ccu_addr  & 0x7F) << 14 ) | 
	  static_cast<uint32_t>( (ccu_chan  & 0xFF) <<  6 ) | 
	  static_cast<uint32_t>( (channel   & 0x3F) <<  0 ) );
}

// -----------------------------------------------------------------------------
//
const SiStripControlKey::ControlPath& SiStripControlKey::path( uint32_t key ) {
  static ControlPath path;
  path.fecCrate_ = (key>>30) & 0x03;
  path.fecSlot_  = (key>>25) & 0x1F;
  path.fecRing_  = (key>>21) & 0x0F;
  path.ccuAddr_  = (key>>14) & 0x7F;
  path.ccuChan_  = (key>> 6) & 0xFF;
  path.channel_  = (key>> 0) & 0x3F;
  if ( path.fecCrate_ == 0x03 ) { path.fecCrate_ = all_; } 
  if ( path.fecSlot_  == 0x1F ) { path.fecSlot_  = all_; } 
  if ( path.fecRing_  == 0x0F ) { path.fecRing_  = all_; } 
  if ( path.ccuAddr_  == 0x7F ) { path.ccuAddr_  = all_; } 
  if ( path.ccuChan_  == 0xFF ) { path.ccuChan_  = all_; } 
  if ( path.channel_  == 0x3F ) { path.channel_  = all_; } 
  return path;  
}

// // -----------------------------------------------------------------------------
// //
// uint32_t SiStripControlKey::key( uint16_t fec_crate,
// 				 uint16_t fec_slot,
// 				 uint16_t fec_ring,
// 				 uint16_t ccu_addr,
// 				 uint16_t ccu_chan,
// 				 uint16_t lld_chan ) {
//   return( static_cast<uint32_t>( (fec_crate & 0x03) << 30 ) | 
// 	  static_cast<uint32_t>( (fec_slot  & 0xFF) << 22 ) | 
// 	  static_cast<uint32_t>( (fec_ring  & 0x0F) << 18 ) | 
// 	  static_cast<uint32_t>( (ccu_addr  & 0xFF) << 10 ) | 
// 	  static_cast<uint32_t>( (ccu_chan  & 0xFF) <<  2 ) | 
// 	  static_cast<uint32_t>( (lld_chan  & 0x03) <<  0 ) );
// }

// // -----------------------------------------------------------------------------
// //
// SiStripControlKey::ControlPath SiStripControlKey::path( uint32_t key ) {
//   ControlPath path;
//   path.fecCrate_ = (key>>30) & 0x03;
//   path.fecSlot_  = (key>>22) & 0xFF;
//   path.fecRing_  = (key>>18) & 0x0F;
//   path.ccuAddr_  = (key>>10) & 0xFF;
//   path.ccuChan_  = (key>> 2) & 0xFF;
//   path.lldChan_  = (key>> 0) & 0x03;
//   if ( path.fecCrate_ == 0x03 ) { path.fecCrate_ = all_; } 
//   if ( path.fecSlot_  == 0xFF ) { path.fecSlot_  = all_; } 
//   if ( path.fecRing_  == 0x0F ) { path.fecRing_  = all_; } 
//   if ( path.ccuAddr_  == 0xFF ) { path.ccuAddr_  = all_; } 
//   if ( path.ccuChan_  == 0xFF ) { path.ccuChan_  = all_; } 
//   if ( path.lldChan_  == 0x03 ) { path.lldChan_  = all_; } 
//   return path;  
// }

