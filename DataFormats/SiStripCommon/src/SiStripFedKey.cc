// Last commit: $Id: SiStripFedKey.cc,v 1.17 2012/07/04 19:04:52 eulisse Exp $

#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/SiStripCommon/interface/Constants.h" 
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForDqm.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForView.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include <iomanip>

// -----------------------------------------------------------------------------
// 
SiStripFedKey::SiStripFedKey( const uint16_t& fed_id,
			      const uint16_t& fe_unit,
			      const uint16_t& fe_chan,
			      const uint16_t& fed_apv ) :
  SiStripKey(),
  fedId_(fed_id), 
  feUnit_(fe_unit), 
  feChan_(fe_chan),
  fedApv_(fed_apv)
{
  // order is important!
  initFromValue();
  initFromKey();
  initFromPath();
  initGranularity();
}

// -----------------------------------------------------------------------------
// 
SiStripFedKey::SiStripFedKey( const uint32_t& fed_key ) :
  SiStripKey(fed_key),
  fedId_(sistrip::invalid_), 
  feUnit_(sistrip::invalid_), 
  feChan_(sistrip::invalid_),
  fedApv_(sistrip::invalid_)
{
  // order is important!
  initFromKey(); 
  initFromValue();
  initFromPath();
  initGranularity();
}

// -----------------------------------------------------------------------------
// 
SiStripFedKey::SiStripFedKey( const std::string& path ) :
  SiStripKey(path),
  fedId_(sistrip::invalid_), 
  feUnit_(sistrip::invalid_), 
  feChan_(sistrip::invalid_),
  fedApv_(sistrip::invalid_)
{
  // order is important!
  initFromPath();
  initFromValue();
  initFromKey();
  initGranularity();
}

// -----------------------------------------------------------------------------
// 
SiStripFedKey::SiStripFedKey( const SiStripFedKey& input ) :
  SiStripKey(),
  fedId_(input.fedId()), 
  feUnit_(input.feUnit()), 
  feChan_(input.feChan()),
  fedApv_(input.fedApv())
{
  key(input.key());
  path(input.path());
  granularity(input.granularity());
}

// -----------------------------------------------------------------------------
// 
SiStripFedKey::SiStripFedKey( const SiStripKey& input ) :
  SiStripKey(),
  fedId_(sistrip::invalid_),
  feUnit_(sistrip::invalid_),
  feChan_(sistrip::invalid_),
  fedApv_(sistrip::invalid_)
{
  const SiStripFedKey& fed_key = dynamic_cast<const SiStripFedKey&>(input);
  if ( (&fed_key) ) {
    key(fed_key.key());
    path(fed_key.path());
    granularity(fed_key.granularity());
    fedId_ = fed_key.fedId(); 
    feUnit_ = fed_key.feUnit(); 
    feChan_ = fed_key.feChan();
    fedApv_ = fed_key.fedApv();
  }
}

// -----------------------------------------------------------------------------
// 
SiStripFedKey::SiStripFedKey() : 
  SiStripKey(),
  fedId_(sistrip::invalid_),
  feUnit_(sistrip::invalid_),
  feChan_(sistrip::invalid_),
  fedApv_(sistrip::invalid_)
{;}

// -----------------------------------------------------------------------------
//
uint16_t SiStripFedKey::fedCh( const uint16_t& fe_unit,
			       const uint16_t& fe_chan ) {
  if ( fe_unit <= sistrip::FEUNITS_PER_FED &&
       fe_chan <= sistrip::FEDCH_PER_FEUNIT ) {
    if ( fe_unit != 0 && fe_chan != 0 ) { 
      return ( 95 - ( 12 * (fe_unit-1) + (fe_chan-1) ) ); 
    } 
  } 
  return sistrip::invalid_;  
}

// -----------------------------------------------------------------------------
//
uint16_t SiStripFedKey::feUnit( const uint16_t& fed_ch ) {
  if ( fed_ch < sistrip::FEDCH_PER_FED ) {
    return ( (95-fed_ch)/12 + 1 );
  }
  return sistrip::invalid_;
}

// -----------------------------------------------------------------------------
//
uint16_t SiStripFedKey::feChan( const uint16_t& fed_ch ) {
  if ( fed_ch < sistrip::FEDCH_PER_FED ) {
    return ( (95-fed_ch)%12 + 1 );
  } 
  return sistrip::invalid_;
}

// -----------------------------------------------------------------------------
//
uint32_t SiStripFedKey::fedIndex( const uint16_t& fed_id,
				  const uint16_t& fed_ch ) {
  if ( fed_id < sistrip::FED_ID_MIN ||
       fed_id > sistrip::FED_ID_MAX ||
       fed_ch >= sistrip::FEDCH_PER_FED ) { return sistrip::invalid32_; }
  return ( fed_id * sistrip::FEDCH_PER_FED + fed_ch );
}

// -----------------------------------------------------------------------------
// 
bool SiStripFedKey::isEqual( const SiStripKey& key ) const {
  const SiStripFedKey& input = dynamic_cast<const SiStripFedKey&>(key);
  if ( !(&input) ) { return false; }
  if ( fedId_ == input.fedId() &&
       feUnit_ == input.feUnit() &&
       feChan_ == input.feChan() &&
       fedApv_ == input.fedApv() ) { 
    return true;
  } else { return false; }
}

// -----------------------------------------------------------------------------
// 
bool SiStripFedKey::isConsistent( const SiStripKey& key ) const {
  const SiStripFedKey& input = dynamic_cast<const SiStripFedKey&>(key);
  if ( !(&input) ) { return false; }
  if ( isEqual(input) ) { return true; }
  else if ( ( fedId_ == 0 || input.fedId() == 0 ) &&
	    ( feUnit_ == 0 || input.feUnit() == 0 ) &&
	    ( feChan_ == 0 || input.feChan() == 0 ) &&
	    ( fedApv_ == 0 || input.fedApv() == 0 ) ) { 
    return true;
  } else { return false; }
}

// -----------------------------------------------------------------------------
//
bool SiStripFedKey::isValid() const { 
  return isValid(sistrip::FED_APV); 
}

// -----------------------------------------------------------------------------
//
bool SiStripFedKey::isValid( const sistrip::Granularity& gran ) const {
  if ( gran == sistrip::FED_SYSTEM ) { return true; }
  else if ( gran == sistrip::UNDEFINED_GRAN ||
	    gran == sistrip::UNKNOWN_GRAN ) { return false; }

  if ( fedId_ != sistrip::invalid_ ) {
    if ( gran == sistrip::FE_DRIVER ) { return true; }
    if ( feUnit_ != sistrip::invalid_ ) {
      if ( gran == sistrip::FE_UNIT ) { return true; }
      if ( feChan_ != sistrip::invalid_ ) {
	if ( gran == sistrip::FE_CHAN ) { return true; }
	if ( fedApv_ != sistrip::invalid_ ) {
	  if ( gran == sistrip::FED_APV ) { return true; }
	}
      }
    }
  }
  return false;
}

// -----------------------------------------------------------------------------
//
bool SiStripFedKey::isInvalid() const { 
  return isInvalid(sistrip::FED_APV); 
}

// -----------------------------------------------------------------------------
//
bool SiStripFedKey::isInvalid( const sistrip::Granularity& gran ) const {
  if ( gran == sistrip::FED_SYSTEM ) { return false; }
  else if ( gran == sistrip::UNDEFINED_GRAN ||
	    gran == sistrip::UNKNOWN_GRAN ) { return false; }

  if ( fedId_ == sistrip::invalid_ ) {
    if ( gran == sistrip::FE_DRIVER ) { return true; }
    if ( feUnit_ == sistrip::invalid_ ) {
      if ( gran == sistrip::FE_UNIT ) { return true; }
      if ( feChan_ == sistrip::invalid_ ) {
	if ( gran == sistrip::FE_CHAN ) { return true; }
	if ( fedApv_ == sistrip::invalid_ ) {
	  if ( gran == sistrip::FED_APV ) { return true; }
	}
      }
    }
  }
  return false;
}

// -----------------------------------------------------------------------------
// 
void SiStripFedKey::initFromValue() {
  
  if ( fedId_ >= sistrip::FED_ID_MIN &&
       fedId_ <= sistrip::FED_ID_MAX ) { 
    fedId_ = fedId_; 
  } else if ( fedId_ == 0 ) { 
    fedId_ = fedId_; 
  } else { 
    fedId_ = sistrip::invalid_; 
  }
  
  if ( feUnit_ <= sistrip::FEUNITS_PER_FED ) { feUnit_ = feUnit_; }
  else { feUnit_ = sistrip::invalid_; }

  if ( feChan_ <= sistrip::FEDCH_PER_FEUNIT ) { feChan_ = feChan_; }
  else { feChan_ = sistrip::invalid_; }
  
  if ( fedApv_ <= sistrip::APVS_PER_FEDCH ) { fedApv_ = fedApv_; }
  else { fedApv_ = sistrip::invalid_; }
  
}

// -----------------------------------------------------------------------------
//
void SiStripFedKey::initFromKey() {

  if ( key() == sistrip::invalid32_ ) { 
    
    // ---------- Set FedKey based on member data ----------
    
    // Initialise to null value
    key(0);
    
    // Extract FED id
    if ( fedId_ >= sistrip::FED_ID_MIN && 
	 fedId_ <= sistrip::FED_ID_MAX ) { 
      key( key() | (fedId_<<fedIdOffset_) );
    } else if ( fedId_ == 0 ) {
      key( key() | (fedId_<<fedIdOffset_) );
    } else {
      key( key() | (fedIdMask_<<fedIdOffset_) ); 
    }
    
    // Extract FE unit
    if ( feUnit_ >= 1 && 
	 feUnit_ <= sistrip::FEUNITS_PER_FED ) { 
      key( key() | (feUnit_<<feUnitOffset_) );
    } else if ( feUnit_ == 0 ) {
      key( key() | (feUnit_<<feUnitOffset_) );
    } else {
      key( key() | (feUnitMask_<<feUnitOffset_) ); 
    }
  
    // Extract FE chan
    if ( feChan_ >= 1 && 
	 feChan_ <= sistrip::FEDCH_PER_FEUNIT ) { 
      key( key() | (feChan_<<feChanOffset_) );
    } else if ( feChan_ == 0 ) {
      key( key() | (feChan_<<feChanOffset_) );
    } else {
      key( key() | (feChanMask_<<feChanOffset_) ); 
    }

    // Extract FED APV
    if ( fedApv_ >= 1 && 
	 fedApv_ <=  sistrip::APVS_PER_FEDCH ) { 
      key( key() | (fedApv_<<fedApvOffset_) );
    } else if ( fedApv_ == 0 ) {
      key( key() | (fedApv_<<fedApvOffset_) );
    } else {
      key( key() | (fedApvMask_<<fedApvOffset_) ); 
    }
    
    // Set invalid key if necessary
    if ( fedId_ == sistrip::invalid_ &&
	 feUnit_ == sistrip::invalid_ &&
	 feChan_ == sistrip::invalid_ &&
	 fedApv_ == sistrip::invalid_ ) { 
      key( sistrip::invalid32_ ); 
    }
    
  } else {
    
    // ---------- Set member data based on FED key ----------
    
    fedId_  = ( key()>>fedIdOffset_ )  & fedIdMask_;
    feUnit_ = ( key()>>feUnitOffset_ ) & feUnitMask_;
    feChan_ = ( key()>>feChanOffset_ ) & feChanMask_;
    fedApv_ = ( key()>>fedApvOffset_ ) & fedApvMask_;
    
    if ( fedId_ == fedIdMask_ )   { fedId_ = sistrip::invalid_; } 
    if ( feUnit_ == feUnitMask_ ) { feUnit_ = sistrip::invalid_; } 
    if ( feChan_ == feChanMask_ ) { feChan_ = sistrip::invalid_; } 
    if ( fedApv_ == fedApvMask_ ) { fedApv_ = sistrip::invalid_; } 

  }
  
}

// -----------------------------------------------------------------------------
// 
void SiStripFedKey::initFromPath() {
  
  if ( path() == sistrip::null_ ) {
    
    // ---------- Set directory path based on member data ----------
      
    std::stringstream dir;
    
    dir << sistrip::root_ << sistrip::dir_ 
	<< sistrip::readoutView_ << sistrip::dir_;
    
    // Add FED id
    if ( fedId_ ) {
      dir << sistrip::feDriver_ << fedId_ << sistrip::dir_;

      // Add FE unit
      if ( feUnit_ ) {
	dir << sistrip::feUnit_ << feUnit_ << sistrip::dir_;

	// Add FE channel
	if ( feChan_ ) {
	  dir << sistrip::feChan_ << feChan_ << sistrip::dir_;

	  // Add FED APV
	  if ( fedApv_ ) {
	    dir << sistrip::fedApv_ << fedApv_ << sistrip::dir_;
	  }
	}
      }
    }
    
    std::string temp( dir.str() );
    path( temp );
    
  } else {
    
    // ---------- Set member data based on directory path ----------
    
    fedId_  = 0;
    feUnit_ = 0;
    feChan_ = 0;
    fedApv_ = 0;
    
    // Check if root is found
    if ( path().find( sistrip::root_ ) == std::string::npos ) {
      std::string temp = path();
      path( std::string(sistrip::root_) + sistrip::dir_ + temp );
    }
    
    size_t curr = 0; // current string position
    size_t next = 0; // next string position
    next = path().find( sistrip::readoutView_, curr );
    
    // Extract view 
    curr = next;
    if ( curr != std::string::npos ) { 
      next = path().find( sistrip::feDriver_, curr );
      std::string readout_view( path(), 
				curr+(sizeof(sistrip::readoutView_) - 1), 
				(next-(sizeof(sistrip::dir_) - 1))-curr );
      
      // Extract FED id
      curr = next;
      if ( curr != std::string::npos ) { 
	next = path().find( sistrip::feUnit_, curr );
	std::string fed_id( path(), 
			    curr+(sizeof(sistrip::feDriver_) - 1), 
			    (next-(sizeof(sistrip::dir_) - 1))-curr );
	fedId_ = atoi( fed_id.c_str() );
	
	// Extract FE unit
	curr = next;
	if ( curr != std::string::npos ) { 
	  next = path().find( sistrip::feChan_, curr );
	  std::string fe_unit( path(), 
			       curr+(sizeof(sistrip::feUnit_) - 1), 
			       next-curr );
	  feUnit_ = atoi( fe_unit.c_str() );
	  
	  // Extract FE channel
	  curr = next;
	  if ( curr != std::string::npos ) { 
	    next = path().find( sistrip::fedApv_, curr );
	    std::string fe_chan( path(), 
				 curr+(sizeof(sistrip::feChan_) - 1), 
				 next-curr );
	    feChan_ = atoi( fe_chan.c_str() );
	    
	    // Extract FED APV 
	    curr = next;
	    if ( curr != std::string::npos ) { 
	      next = std::string::npos;
	      std::string fed_apv( path(), 
				   curr+(sizeof(sistrip::fedApv_) - 1), 
				   next-curr );
	      fedApv_ = atoi( fed_apv.c_str() );
	    }
	  }
	}
      }
    } else {
      std::stringstream ss;
      ss << sistrip::root_ << sistrip::dir_;
      //ss << sistrip::root_ << sistrip::dir_ 
      //<< sistrip::unknownView_ << sistrip::dir_;
      std::string temp( ss.str() );
      path( temp );
    }

  }
  
}

// -----------------------------------------------------------------------------
// 
void SiStripFedKey::initGranularity() {
  
  granularity( sistrip::FED_SYSTEM );
  channel(0);
  if ( fedId_ && fedId_ != sistrip::invalid_ ) {
    granularity( sistrip::FE_DRIVER ); 
    channel(fedId_);
    if ( feUnit_ && feUnit_ != sistrip::invalid_ ) {
      granularity( sistrip::FE_UNIT );
      channel(feUnit_);
      if ( feChan_ && feChan_ != sistrip::invalid_ ) {
	granularity( sistrip::FE_CHAN );
	channel(feChan_);
	if ( fedApv_ && fedApv_ != sistrip::invalid_ ) {
	  granularity( sistrip::FED_APV );
	  channel(fedApv_);
	} else if ( fedApv_ == sistrip::invalid_ ) { 
	  granularity( sistrip::UNKNOWN_GRAN ); 
	  channel(sistrip::invalid_);
	}
      } else if ( feChan_ == sistrip::invalid_ ) { 
	granularity( sistrip::UNKNOWN_GRAN ); 
	channel(sistrip::invalid_);
      }
    } else if ( feUnit_ == sistrip::invalid_ ) { 
      granularity( sistrip::UNKNOWN_GRAN ); 
      channel(sistrip::invalid_);
    }
  } else if ( fedId_ == sistrip::invalid_ ) { 
    granularity( sistrip::UNKNOWN_GRAN ); 
    channel(sistrip::invalid_);
  }
  
}

// -----------------------------------------------------------------------------
//
void SiStripFedKey::terse( std::stringstream& ss ) const {
  ss << "FED:crate/slot/id/unit/chan/apv= "
     << "-" << "/"
     << "-" << "/"
     << fedId() << "/"
     << feUnit() << "/"
     << feChan() << "/"
     << fedApv();
//   ss << " FedKey"
//     //<<"=0x" 
//     //<< std::hex
//     //<< std::setfill('0') << std::setw(8) << key() << std::setfill(' ') 
//     //<< std::dec
//     //<< ", " << ( isValid() ? "Valid" : "Invalid" )
//     //<< ", FedCrate=" << fedCrate()
//     //<< ", FedSlot=" << fedSlot()
//      << ", FedId=" << fedId()
//      << ", FeUnit=" << feUnit()
//      << ", FeChan=" << feChan()
//     //<< ", FedChannel=" << fedChannel()
//      << ", FedApv=" << fedApv();
}

// -----------------------------------------------------------------------------
//
void SiStripFedKey::print( std::stringstream& ss ) const {
  ss << " [SiStripFedKey::print]" << std::endl
     << std::hex
     << " FED key         : 0x" 
     << std::setfill('0') 
     << std::setw(8) << key() << std::endl
     << std::setfill(' ') 
     << std::dec
     << " FED id          : " << fedId() << std::endl
     << " Front-End unit  : " << feUnit() << std::endl 
     << " Front-End chan  : " << feChan() << std::endl
     << " (internal chan) : " 
     << "(" << fedChannel() << ")" << std::endl
     << " FED APV         : " << fedApv() << std::endl
     << " Directory       : " << path() << std::endl
     << " Granularity     : "
     << SiStripEnumsAndStrings::granularity( granularity() ) << std::endl
     << " Channel         : " << channel() << std::endl
     << " isValid         : " << isValid();
}

// -----------------------------------------------------------------------------
//
std::ostream& operator<< ( std::ostream& os, const SiStripFedKey& input ) {
  std::stringstream ss;
  input.print(ss);
  os << ss.str();
  return os;
}
