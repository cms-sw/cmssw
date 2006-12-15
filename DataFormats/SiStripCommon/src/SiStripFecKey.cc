#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include <iomanip>

// -----------------------------------------------------------------------------
//
SiStripFecKey::Path::Path() 
  : fecCrate_(sistrip::invalid_), 
    fecSlot_(sistrip::invalid_),
    fecRing_(sistrip::invalid_), 
    ccuAddr_(sistrip::invalid_),
    ccuChan_(sistrip::invalid_), 
    channel_(sistrip::invalid_) {;}

// -----------------------------------------------------------------------------
//
/*
  FEC crate [1-4]:     3-bits, 0=all, 0x07=invalid
  FEC slot [1-21]:     5-bits, 0=all, 0x1F=invalid
  FEC ring [1-8]:      4-bits, 0=all, 0x0F=invalid
  CCU module [1-127]:  8-bits, 0=all, 0xFF=invalid
  FE module [16-31]:   5-bits, 0=all, 0x1F=invalid
  LLD/APV [1-3,32-37]: 7-bits, 0=all, 0x7F=invalid
*/
SiStripFecKey::Path::Path( const uint16_t& fec_crate, 
			   const uint16_t& fec_slot, 
			   const uint16_t& fec_ring, 
			   const uint16_t& ccu_addr, 
			   const uint16_t& ccu_chan,
			   const uint16_t& channel ) 
  : fecCrate_(fec_crate), 
    fecSlot_(fec_slot),
    fecRing_(fec_ring), 
    ccuAddr_(ccu_addr),
    ccuChan_(ccu_chan), 
    channel_(channel) {;}

// -----------------------------------------------------------------------------
// 
bool SiStripFecKey::Path::isEqual( const Path& input ) const {
  if ( fecCrate_ == input.fecCrate_ &&
       fecSlot_ == input.fecSlot_ &&
       fecRing_ == input.fecRing_ &&
       ccuAddr_ == input.ccuAddr_ &&
       ccuChan_ == input.ccuChan_ &&
       channel_ == input.channel_ ) { 
    return true;
  } else { return false; }
}

// -----------------------------------------------------------------------------
// 
bool SiStripFecKey::Path::isConsistent( const Path& input ) const {
  if ( isEqual(input) ) { return true; }
  else if ( ( fecCrate_ == 0 || input.fecCrate_ == 0 ) &&
	    ( fecSlot_ == 0 || input.fecSlot_ == 0 ) &&
	    ( fecRing_ == 0 || input.fecRing_ == 0 ) &&
	    ( ccuAddr_ == 0 || input.ccuAddr_ == 0 ) &&
	    ( channel_ == 0 || input.channel_ == 0 ) ) {
    return true;
  } else { return false; }
}

// -----------------------------------------------------------------------------
//
bool SiStripFecKey::Path::isInvalid() const {
  if ( fecCrate_ == sistrip::invalid_ &&
       fecSlot_ == sistrip::invalid_ &&
       fecRing_ == sistrip::invalid_ &&
       ccuAddr_ == sistrip::invalid_ &&
       ccuChan_ == sistrip::invalid_ &&
       channel_ == sistrip::invalid_ ) {
    return true;
  } else { return false; }
}

// -----------------------------------------------------------------------------
//
bool SiStripFecKey::Path::isInvalid( const sistrip::Granularity& gran ) const {
  if ( fecCrate_ == sistrip::invalid_ ) {
    if ( gran == sistrip::FEC_CRATE ) { return true; }
    if ( fecSlot_ == sistrip::invalid_ ) {
      if ( gran == sistrip::FEC_RING ) { return true; }
      if ( fecRing_ == sistrip::invalid_ ) {
	if ( gran == sistrip::FEC_RING ) { return true; }
	if ( ccuAddr_ == sistrip::invalid_ ) {
	  if ( gran == sistrip::CCU_ADDR ) { return true; }
	  if ( ccuChan_ == sistrip::invalid_ ) {
	    if ( gran == sistrip::CCU_CHAN ) { return true; }
	    if ( channel_ == sistrip::invalid_ ) {
	      if ( gran == sistrip::LLD_CHAN  || 
		   gran == sistrip::APV ) { return true; }
	    }
	  }
	}
      }
    }
  }
  return false;
}

// -----------------------------------------------------------------------------
//
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
bool SiStripFecKey::isEqual( const uint32_t& key1, 
			     const uint32_t& key2 ) {
  SiStripFecKey::Path path1 = SiStripFecKey::path( key1 ) ;
  SiStripFecKey::Path path2 = SiStripFecKey::path( key2 ) ;
  return path1.isEqual( path2 );
}

// -----------------------------------------------------------------------------
//
bool SiStripFecKey::isConsistent( const uint32_t& key1, 
				  const uint32_t& key2 ) {
  SiStripFecKey::Path path1 = SiStripFecKey::path( key1 ) ;
  SiStripFecKey::Path path2 = SiStripFecKey::path( key2 ) ;
  return path1.isConsistent( path2 );
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









