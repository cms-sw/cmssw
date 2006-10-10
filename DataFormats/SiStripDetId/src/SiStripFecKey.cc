#include "DataFormats/SiStripDetId/interface/SiStripFecKey.h"
#include <iomanip>

//@@ ALTERNATIVE METHOD?...  
// can use 8 bits for device address, start crate/slot/ring numbering
// from zero and use only slot numbers 0-15 in crate, to gain 3 bits

// -----------------------------------------------------------------------------
// 
uint32_t SiStripFecKey::key( uint16_t fec_crate,
			     uint16_t fec_slot,
			     uint16_t fec_ring,
			     uint16_t ccu_addr,
			     uint16_t ccu_chan,
			     uint16_t channel ) {
  return( static_cast<uint32_t>( ( fec_crate & SiStripFecKey::fecCrateMask_ ) << SiStripFecKey::fecCrateOffset_ ) | 
	  static_cast<uint32_t>( ( fec_slot  & SiStripFecKey::fecSlotMask_  ) << SiStripFecKey::fecSlotOffset_  ) | 
	  static_cast<uint32_t>( ( fec_ring  & SiStripFecKey::fecRingMask_  ) << SiStripFecKey::fecRingOffset_  ) | 
	  static_cast<uint32_t>( ( ccu_addr  & SiStripFecKey::ccuAddrMask_  ) << SiStripFecKey::ccuAddrOffset_  ) | 
	  static_cast<uint32_t>( ( ccu_chan  & SiStripFecKey::ccuChanMask_  ) << SiStripFecKey::ccuChanOffset_  ) | 
	  static_cast<uint32_t>( ( channel   & SiStripFecKey::channelMask_  ) << SiStripFecKey::channelOffset_  ) );
}

// -----------------------------------------------------------------------------
// 
uint32_t SiStripFecKey::key( const Path& path ) {
  return SiStripFecKey::key( path.fecCrate_,
			     path.fecSlot_,
			     path.fecRing_,
			     path.ccuAddr_,
			     path.ccuChan_,
			     path.channel_ );
}

// -----------------------------------------------------------------------------
//
SiStripFecKey::Path SiStripFecKey::path( uint32_t key ) {
  Path tmp;
  tmp.fecCrate_ = ( key>>SiStripFecKey::fecCrateOffset_ ) & SiStripFecKey::fecCrateMask_;
  tmp.fecSlot_  = ( key>>SiStripFecKey::fecSlotOffset_  ) & SiStripFecKey::fecSlotMask_;
  tmp.fecRing_  = ( key>>SiStripFecKey::fecRingOffset_  ) & SiStripFecKey::fecRingMask_;
  tmp.ccuAddr_  = ( key>>SiStripFecKey::ccuAddrOffset_  ) & SiStripFecKey::ccuAddrMask_;
  tmp.ccuChan_  = ( key>>SiStripFecKey::ccuChanOffset_  ) & SiStripFecKey::ccuChanMask_;
  tmp.channel_  = ( key>>SiStripFecKey::channelOffset_  ) & SiStripFecKey::channelMask_;
  if ( tmp.fecCrate_ == SiStripFecKey::fecCrateMask_ ) { tmp.fecCrate_ = sistrip::invalid_; } 
  if ( tmp.fecSlot_  == SiStripFecKey::fecSlotMask_ )  { tmp.fecSlot_  = sistrip::invalid_; } 
  if ( tmp.fecRing_  == SiStripFecKey::fecRingMask_ )  { tmp.fecRing_  = sistrip::invalid_; } 
  if ( tmp.ccuAddr_  == SiStripFecKey::ccuAddrMask_ )  { tmp.ccuAddr_  = sistrip::invalid_; } 
  if ( tmp.ccuChan_  == SiStripFecKey::ccuChanMask_ )  { tmp.ccuChan_  = sistrip::invalid_; } 
  if ( tmp.channel_  == SiStripFecKey::channelMask_ )  { tmp.channel_  = sistrip::invalid_; } 
  return tmp;  
}

// -----------------------------------------------------------------------------
//
std::ostream& operator<< ( std::ostream& os, const SiStripFecKey::Path& path ) {
  return os << "[SiStripFecKey::Path]"
	    << std::hex
	    << " key=0x" << std::setfill('0') << std::setw(8) << SiStripFecKey::key(path)
	    << std::dec
	    << " fecCrate=" << path.fecCrate_
	    << " fecSlot=" << path.fecSlot_
	    << " fecRing=" << path.fecRing_
	    << " ccuAddr=" << path.ccuAddr_
	    << " ccuChan=" << path.ccuChan_
	    << " channel=" << path.channel_;
}









// // -----------------------------------------------------------------------------
// //
// uint32_t SiStripFecKey::key( uint16_t fec_crate,
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
// SiStripFecKey::ControlPath SiStripFecKey::path( uint32_t key ) {
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

