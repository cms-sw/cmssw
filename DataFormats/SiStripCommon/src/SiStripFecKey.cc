#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include <iomanip>

// -----------------------------------------------------------------------------
//
/*
  FEC crate: 3-bits, value=1->4,       0=all, 0x07=invalid
  FEC slot:  5-bits, value=1->21,      0=all, 0x1F=invalid
  FEC ring:  4-bits, value=1->8,       0=all, 0x0F=invalid
  CCU addr:  8-bits, value=1->127,     0=all, 0xFF=invalid
  CCU chan:  5-bits, value=16->31,     0=all, 0x1F=invalid
  channel:   7-bits, value=1->3,32-37, 0=all, 0x7F=invalid
*/
uint32_t SiStripFecKey::key( uint16_t fec_crate,
			     uint16_t fec_slot,
			     uint16_t fec_ring,
			     uint16_t ccu_addr,
			     uint16_t ccu_chan,
			     uint16_t channel ) {

  uint32_t temp = 0;

  if ( fec_crate > sistrip::FEC_CRATE_MAX ) { temp |= (fecCrateMask_<<fecCrateOffset_); }
  else { temp |= (fec_crate<<fecCrateOffset_); }

  if ( fec_slot > sistrip::CRATE_SLOT_MAX ) { temp |= (fecSlotMask_<<fecSlotOffset_); } 
  else { temp |= (fec_slot<<fecSlotOffset_); }

  if ( fec_ring > sistrip::FEC_RING_MAX ) { temp |= (fecRingMask_<<fecRingOffset_); }
  else { temp |= (fec_ring<<fecRingOffset_); }

  if ( ccu_addr > sistrip::CCU_ADDR_MAX ) { temp |= (ccuAddrMask_<<ccuAddrOffset_); }
  else { temp |= (ccu_addr<<ccuAddrOffset_); 
  }

  if ( ccu_chan >= sistrip::CCU_CHAN_MIN &&
       ccu_chan <= sistrip::CCU_CHAN_MAX ) { 
    temp |= ( (ccu_chan-(sistrip::CCU_CHAN_MIN-1)) << ccuChanOffset_ ); 
  } else if ( !ccu_chan ) {
    temp |= ( ccu_chan << ccuChanOffset_ ); 
  } else { 
    temp |= ( ccuChanMask_ << ccuChanOffset_ ); 
  }

  if ( channel >= sistrip::LLD_CHAN_MIN && 
       channel <= sistrip::LLD_CHAN_MAX ) { 
    temp |= ( (channel-sistrip::LLD_CHAN_MIN+1) << channelOffset_ ); 
  } else if ( channel >= sistrip::APV_I2C_MIN && 
	      channel <= sistrip::APV_I2C_MAX  ) { 
    temp |= ( (channel-(sistrip::APV_I2C_MIN-sistrip::LLD_CHAN_MAX-1)) << channelOffset_ ); 
  } else if ( !channel ) {
    temp |= ( channel << channelOffset_ ); 
  } else {
    temp |= ( channelMask_ << channelOffset_ ); 
  }
  
  return temp;

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

  tmp.fecCrate_ = ( key>>fecCrateOffset_ ) & fecCrateMask_;
  tmp.fecSlot_ = ( key>>fecSlotOffset_ ) & fecSlotMask_;
  tmp.fecRing_ = ( key>>fecRingOffset_ ) & fecRingMask_;
  tmp.ccuAddr_ = ( key>>ccuAddrOffset_ ) & ccuAddrMask_;
  tmp.ccuChan_ = ( key>>ccuChanOffset_ ) & ccuChanMask_;
  tmp.channel_ = ( key>>channelOffset_ ) & channelMask_;

  if ( tmp.fecCrate_ == fecCrateMask_ ) { tmp.fecCrate_ = sistrip::invalid_; } 
  if ( tmp.fecSlot_ == fecSlotMask_ ) { tmp.fecSlot_ = sistrip::invalid_; } 
  if ( tmp.fecRing_ == fecRingMask_ ) { tmp.fecRing_ = sistrip::invalid_; } 
  if ( tmp.ccuAddr_ == ccuAddrMask_ ) { tmp.ccuAddr_ = sistrip::invalid_; } 

  if ( tmp.ccuChan_ == ccuChanMask_ ) { 
    tmp.ccuChan_ = sistrip::invalid_; 
  } else if ( tmp.ccuChan_ ) { 
    tmp.ccuChan_ += (sistrip::CCU_CHAN_MIN-1); 
  } 

  if ( tmp.channel_ == channelMask_ ) { 
    tmp.channel_ = sistrip::invalid_; 
  } else if ( tmp.channel_ >= sistrip::LLD_CHAN_MIN && 
	      tmp.channel_ <= sistrip::LLD_CHAN_MAX ) { 
    tmp.channel_ += (sistrip::LLD_CHAN_MIN-1); 
  } else if ( tmp.channel_ >= (sistrip::LLD_CHAN_MAX+1) && 
	      tmp.channel_ <= (sistrip::LLD_CHAN_MAX+1+sistrip::APV_I2C_MAX-sistrip::APV_I2C_MIN) ) { 
    tmp.channel_ += (sistrip::APV_I2C_MIN-sistrip::LLD_CHAN_MAX-1); 
  }

  return tmp;  
}

// -----------------------------------------------------------------------------
//
std::ostream& operator<< ( std::ostream& os, const SiStripFecKey::Path& path ) {

  return os << std::hex
	    << " FecKey: 0x" << std::setfill('0') << std::setw(8) << SiStripFecKey::key(path)
	    << std::dec
	    << " Crate/FEC/Ring/CCU/Module/Channel: " 
	    << path.fecCrate_ << "/"
	    << path.fecSlot_ << "/"
	    << path.fecRing_ << "/"
	    << path.ccuAddr_ << "/"
	    << path.ccuChan_ << "/"
	    << path.channel_;

}









