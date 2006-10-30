#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include <iomanip>
#include <iostream>

using namespace std;

// -----------------------------------------------------------------------------
// 
SiStripFedKey::Path::Path() : 
  fedCrate_(sistrip::invalid_),
  fedSlot_(sistrip::invalid_),
  fedId_(sistrip::invalid_),
  fedCh_(sistrip::invalid_),
  fedApv_(sistrip::invalid_),
  feUnit_(sistrip::invalid_),
  feChan_(sistrip::invalid_) 
{
  //@@ nothing here
}

// -----------------------------------------------------------------------------
// 
SiStripFedKey::Path::Path( const uint16_t& fed_id,
			   const uint16_t& fed_ch,
			   const uint16_t& fed_apv ) :
  fedCrate_(sistrip::invalid_),
  fedSlot_(sistrip::invalid_),
  fedId_(fed_id), 
  fedCh_(fed_ch), 
  fedApv_(fed_apv),
  feUnit_(sistrip::invalid_), 
  feChan_(sistrip::invalid_) 
{
  
  if ( fedId_ >= sistrip::FED_ID_MIN &&
       fedId_ < sistrip::FED_ID_LAST ) {
    fedCrate_ = fedId_ - sistrip::FED_ID_MIN; 
    fedSlot_  = fedCrate_ % sistrip::SLOTS_PER_CRATE;
    fedCrate_ /= sistrip::SLOTS_PER_CRATE;
  } else if ( fedId_ < sistrip::FED_ID_MIN ) { 
    fedCrate_ = sistrip::invalid_;
    fedSlot_  = sistrip::invalid_;
  } else {
    fedId_    = sistrip::invalid_;
    fedCrate_ = sistrip::invalid_;
    fedSlot_  = sistrip::invalid_;
  }
  
  if ( fedCh_ < sistrip::FEDCH_PER_FED ) {
    feUnit_ = (fed_ch/12) + 1;
    feChan_ = (fed_ch%12) + 1;
  } else {
    fedCh_  = sistrip::invalid_;
    feUnit_ = sistrip::invalid_;
    feChan_ = sistrip::invalid_;
  }

  if ( fedApv_ > sistrip::APVS_PER_FEDCH ) { 
    fedApv_ = sistrip::invalid_;
  }

}

// -----------------------------------------------------------------------------
// 
SiStripFedKey::Path::Path( const uint16_t& fed_id,
			   const uint16_t& fe_unit,
			   const uint16_t& fe_chan,
			   const uint16_t& fed_apv ) :
  fedCrate_(sistrip::invalid_),
  fedSlot_(sistrip::invalid_),
  fedId_(fed_id), 
  fedCh_(sistrip::invalid_), 
  fedApv_(fed_apv),
  feUnit_(fe_unit), 
  feChan_(fe_chan) 
{

  if ( fedId_ >= sistrip::FED_ID_MIN &&
       fedId_ < sistrip::FED_ID_LAST ) {
    fedCrate_ = fedId_ - sistrip::FED_ID_MIN; 
    fedSlot_  = fedCrate_ % sistrip::SLOTS_PER_CRATE;
    fedCrate_ /= sistrip::SLOTS_PER_CRATE;
  } else if ( fedId_ < sistrip::FED_ID_MIN ) { 
    fedCrate_ = sistrip::invalid_;
    fedSlot_  = sistrip::invalid_;
  } else {
    fedId_    = sistrip::invalid_;
    fedCrate_ = sistrip::invalid_;
    fedSlot_  = sistrip::invalid_;
  }

  if ( feUnit_ <= sistrip::FEUNITS_PER_FED ) {
    if ( feChan_ <= sistrip::FEDCH_PER_FEUNIT ) {
      if ( feUnit_ && feChan_ ) { fedCh_ = 12 * (feUnit_-1) + (feChan_-1); }
      else { fedCh_ = sistrip::invalid_; }
    } else { feChan_ = sistrip::invalid_; } 
  } else { feUnit_ = sistrip::invalid_; }

  if ( fedApv_ > sistrip::APVS_PER_FEDCH ) { 
    fedApv_ = sistrip::invalid_;
  }
  
}

// -----------------------------------------------------------------------------
// 
uint32_t SiStripFedKey::key( const uint16_t& fed_id,
			     const uint16_t& fed_ch,
			     const uint16_t& fed_apv ) {
  Path tmp( fed_id, fed_ch, fed_apv );
  return key( tmp );
}

// -----------------------------------------------------------------------------
// 
uint32_t SiStripFedKey::key( const uint16_t& fed_id,
			     const uint16_t& fe_unit,
			     const uint16_t& fe_chan,
			     const uint16_t& fed_apv ) {
  Path tmp( fed_id, fe_unit, fe_chan, fed_apv );
  return key( tmp );
}

// -----------------------------------------------------------------------------
// 
uint32_t SiStripFedKey::key( const Path& path ) {

  uint32_t temp = 0;
  
  if ( path.fedCrate_ > sistrip::FED_CRATE_MAX ) { temp |= (fedCrateMask_<<fedCrateOffset_); }
  else { temp |= (path.fedCrate_<<fedCrateOffset_); }
  
  if ( path.fedSlot_ > sistrip::CRATE_SLOT_MAX ) { temp |= (fedSlotMask_<<fedSlotOffset_); }
  else { temp |= (path.fedSlot_<<fedSlotOffset_); }
  
  if ( //path.fedId_ >= sistrip::FED_ID_MIN && 
      path.fedId_ < sistrip::FED_ID_LAST ) { 
    temp |= (path.fedId_<<fedIdOffset_);
    // } else if ( !path.fedId_ ) {
    // temp |= (path.fedId_<<fedIdOffset_);
  } else {
    temp |= (fedIdMask_<<fedIdOffset_); 
  }
  
  if ( path.feUnit_ > sistrip::FEUNITS_PER_FED ) { temp |= (feUnitMask_<<feUnitOffset_); }
  else { temp |= (path.feUnit_<<feUnitOffset_); }
  
  if ( path.feChan_ > sistrip::FEDCH_PER_FEUNIT ) { temp |= (feChanMask_<<feChanOffset_); }
  else { temp |= (path.feChan_<<feChanOffset_); }
  
  if ( path.fedApv_ > sistrip::APVS_PER_FEDCH ) { temp |= (fedApvMask_<<fedApvOffset_); }
  else { temp |= (path.fedApv_<<fedApvOffset_); }
  
  return temp;
  
}

// -----------------------------------------------------------------------------
//
SiStripFedKey::Path SiStripFedKey::path( uint32_t key ) {

  Path tmp;

  tmp.fedCrate_ = ( key>>fedCrateOffset_ ) & fedCrateMask_;
  tmp.fedSlot_ = ( key>>fedSlotOffset_ ) & fedSlotMask_;
  tmp.fedId_ = ( key>>fedIdOffset_ ) & fedIdMask_;
  tmp.fedApv_ = ( key>>fedApvOffset_ ) & fedApvMask_;
  tmp.feUnit_ = ( key>>feUnitOffset_ ) & feUnitMask_;
  tmp.feChan_ = ( key>>feChanOffset_ ) & feChanMask_;

  if ( tmp.fedCrate_ == fedCrateMask_ ) { tmp.fedCrate_ = sistrip::invalid_; } 
  if ( tmp.fedSlot_ == fedSlotMask_ ) { tmp.fedSlot_ = sistrip::invalid_; } 
  if ( tmp.fedId_ == fedIdMask_ ) { tmp.fedId_ = sistrip::invalid_; } 
  if ( tmp.fedApv_ == fedApvMask_ ) { tmp.fedApv_ = sistrip::invalid_; } 
  if ( tmp.feUnit_ == feUnitMask_ ) { tmp.feUnit_ = sistrip::invalid_; } 
  if ( tmp.feChan_ == feChanMask_ ) { tmp.feChan_ = sistrip::invalid_; } 
  
  if ( tmp.feUnit_ && tmp.feUnit_ < sistrip::invalid_ &&
       tmp.feChan_ && tmp.feChan_ < sistrip::invalid_ ) {
    tmp.fedCh_ = 12 * (tmp.feUnit_-1) + (tmp.feChan_-1);
  }
  
  return tmp;  
}

// -----------------------------------------------------------------------------
//
std::ostream& operator<< ( std::ostream& os, const SiStripFedKey::Path& path ) {
  return os << std::hex
	    << " FedKey: 0x" << std::setfill('0') << std::setw(8) << SiStripFedKey::key(path)
	    << std::dec
	    << " Crate/Slot: " << path.fedCrate_ << "/" << path.fedSlot_
	    << " FedId/Ch: " << path.fedId_ << "/" << path.fedCh_
	    << " FeUnit/Chan: " << path.feUnit_ << "/" << path.feChan_
	    << " FedApv: " << path.fedApv_;
}
