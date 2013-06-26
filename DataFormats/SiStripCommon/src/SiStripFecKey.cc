// Last commit: $Id: SiStripFecKey.cc,v 1.23 2012/07/04 19:04:51 eulisse Exp $

#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripNullKey.h"
#include "DataFormats/SiStripCommon/interface/Constants.h" 
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForDqm.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForView.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include <iomanip>

// -----------------------------------------------------------------------------
//
SiStripFecKey::SiStripFecKey( const uint16_t& fec_crate, 
			      const uint16_t& fec_slot, 
			      const uint16_t& fec_ring, 
			      const uint16_t& ccu_addr, 
			      const uint16_t& ccu_chan,
			      const uint16_t& lld_chan,
			      const uint16_t& i2c_addr ) :
  SiStripKey(),
  fecCrate_(fec_crate), 
  fecSlot_(fec_slot),
  fecRing_(fec_ring), 
  ccuAddr_(ccu_addr),
  ccuChan_(ccu_chan),
  lldChan_(lld_chan),
  i2cAddr_(i2c_addr)
{
  // order is important!
  initFromValue();
  initFromKey();
  initFromPath();
  initGranularity();
}

// -----------------------------------------------------------------------------
// 
SiStripFecKey::SiStripFecKey( const uint32_t& fec_key ) :
  SiStripKey(fec_key),
  fecCrate_(sistrip::invalid_), 
  fecSlot_(sistrip::invalid_),
  fecRing_(sistrip::invalid_), 
  ccuAddr_(sistrip::invalid_),
  ccuChan_(sistrip::invalid_), 
  lldChan_(sistrip::invalid_),
  i2cAddr_(sistrip::invalid_)
{
  // order is important!
  initFromKey(); 
  initFromValue();
  initFromPath();
  initGranularity();
}

// -----------------------------------------------------------------------------
// 
SiStripFecKey::SiStripFecKey( const std::string& path ) :
  SiStripKey(path),
  fecCrate_(sistrip::invalid_), 
  fecSlot_(sistrip::invalid_),
  fecRing_(sistrip::invalid_), 
  ccuAddr_(sistrip::invalid_),
  ccuChan_(sistrip::invalid_), 
  lldChan_(sistrip::invalid_),
  i2cAddr_(sistrip::invalid_)
{
  // order is important!
  initFromPath();
  initFromValue();
  initFromKey(); 
  initGranularity();
}

// -----------------------------------------------------------------------------
// 
SiStripFecKey::SiStripFecKey( const SiStripFecKey& input ) :
  SiStripKey(),
  fecCrate_(input.fecCrate()), 
  fecSlot_(input.fecSlot()),
  fecRing_(input.fecRing()), 
  ccuAddr_(input.ccuAddr()),
  ccuChan_(input.ccuChan()), 
  lldChan_(input.lldChan()), 
  i2cAddr_(input.i2cAddr())
{
  key(input.key());
  path(input.path());
  granularity(input.granularity());
}

// -----------------------------------------------------------------------------
// 
SiStripFecKey::SiStripFecKey( const SiStripKey& input ) :
  SiStripKey(),
  fecCrate_(sistrip::invalid_), 
  fecSlot_(sistrip::invalid_),
  fecRing_(sistrip::invalid_), 
  ccuAddr_(sistrip::invalid_),
  ccuChan_(sistrip::invalid_), 
  lldChan_(sistrip::invalid_),
  i2cAddr_(sistrip::invalid_)
{
  const SiStripFecKey& fec_key = dynamic_cast<const SiStripFecKey&>(input);
  if ( (&fec_key) ) {
    key(fec_key.key());
    path(fec_key.path());
    granularity(fec_key.granularity());
    fecCrate_ = fec_key.fecCrate(); 
    fecSlot_ = fec_key.fecSlot();
    fecRing_ = fec_key.fecRing(); 
    ccuAddr_ = fec_key.ccuAddr();
    ccuChan_ = fec_key.ccuChan(); 
    lldChan_ = fec_key.lldChan();
    i2cAddr_ = fec_key.i2cAddr();
  }
}

// -----------------------------------------------------------------------------
// 
SiStripFecKey::SiStripFecKey( const SiStripKey& input,
			      const sistrip::Granularity& gran ) :
  SiStripKey(),
  fecCrate_(0), 
  fecSlot_(0),
  fecRing_(0), 
  ccuAddr_(0),
  ccuChan_(0), 
  lldChan_(0),
  i2cAddr_(0)
{
  const SiStripFecKey& fec_key = dynamic_cast<const SiStripFecKey&>(input);
  if ( (&fec_key) ) {
    
    if ( gran == sistrip::FEC_CRATE || gran == sistrip::FEC_SLOT ||
	 gran == sistrip::FEC_RING || gran == sistrip::CCU_ADDR ||
	 gran == sistrip::CCU_CHAN || gran == sistrip::LLD_CHAN ||
	 gran == sistrip::APV ) {
      fecCrate_ = fec_key.fecCrate(); 
    }

    if ( gran == sistrip::FEC_SLOT || gran == sistrip::FEC_RING ||
	 gran == sistrip::CCU_ADDR || gran == sistrip::CCU_CHAN ||
	 gran == sistrip::LLD_CHAN || gran == sistrip::APV ) {
      fecSlot_ = fec_key.fecSlot();
    }

    if ( gran == sistrip::FEC_RING || gran == sistrip::CCU_ADDR ||
	 gran == sistrip::CCU_CHAN || gran == sistrip::LLD_CHAN ||
	 gran == sistrip::APV ) {
      fecRing_ = fec_key.fecRing(); 
    }

    if ( gran == sistrip::CCU_ADDR || gran == sistrip::CCU_CHAN ||
	 gran == sistrip::LLD_CHAN || gran == sistrip::APV ) {
      ccuAddr_ = fec_key.ccuAddr();
    }

    if ( gran == sistrip::CCU_CHAN || gran == sistrip::LLD_CHAN ||
	 gran == sistrip::APV ) {
      ccuChan_ = fec_key.ccuChan(); 
    }

    if ( gran == sistrip::LLD_CHAN || gran == sistrip::APV ) {
      lldChan_ = fec_key.lldChan();
    }

    if ( gran == sistrip::APV ) {
      i2cAddr_ = fec_key.i2cAddr();
    }

    initFromValue();
    initFromKey();
    initFromPath();
    initGranularity();
    
  }

}

// -----------------------------------------------------------------------------
// 
SiStripFecKey::SiStripFecKey() :
  SiStripKey(),
  fecCrate_(sistrip::invalid_), 
  fecSlot_(sistrip::invalid_),
  fecRing_(sistrip::invalid_), 
  ccuAddr_(sistrip::invalid_),
  ccuChan_(sistrip::invalid_), 
  lldChan_(sistrip::invalid_),
  i2cAddr_(sistrip::invalid_)
{;}

// -----------------------------------------------------------------------------
// 
uint16_t SiStripFecKey::hybridPos( const uint16_t& i2c_addr ) {
  if ( i2c_addr < sistrip::APV_I2C_MIN ||
       i2c_addr > sistrip::APV_I2C_MAX ) {
    return sistrip::invalid_;
  }
  return ( i2c_addr - sistrip::APV_I2C_MIN + 1 );
}

// -----------------------------------------------------------------------------
// 
uint16_t SiStripFecKey::i2cAddr( const uint16_t& hybrid_pos ) {
  if ( !hybrid_pos ||
       hybrid_pos > 
       ( sistrip::APV_I2C_MAX - 
	 sistrip::APV_I2C_MIN + 1 ) ) {
    return sistrip::invalid_;
  }
  return ( hybrid_pos + sistrip::APV_I2C_MIN - 1 );
}

// -----------------------------------------------------------------------------
// 
uint16_t SiStripFecKey::i2cAddr( const uint16_t& lld_chan,
				 const bool& first_apv ) {
  if ( lld_chan < sistrip::LLD_CHAN_MIN ||
       lld_chan > sistrip::LLD_CHAN_MAX ) {
    return sistrip::invalid_; 
  }
  return ( sistrip::APV_I2C_MIN + lld_chan * sistrip::APVS_PER_CHAN - (first_apv?2:1) );
}

// -----------------------------------------------------------------------------
// 
uint16_t SiStripFecKey::lldChan( const uint16_t& i2c_addr ) {
  if ( i2c_addr == 0 ) { return 0; }
  else if ( i2c_addr < sistrip::APV_I2C_MIN ||
	    i2c_addr > sistrip::APV_I2C_MAX ) {
    return sistrip::invalid_;
  }
  return ( ( i2c_addr - sistrip::APV_I2C_MIN ) / 2 + 1 );
}

// -----------------------------------------------------------------------------
// 
bool SiStripFecKey::firstApvOfPair( const uint16_t& i2c_addr ) {
  if ( i2c_addr < sistrip::APV_I2C_MIN ||
       i2c_addr > sistrip::APV_I2C_MAX ) {
    return sistrip::invalid_;
  }
  return ( ( ( i2c_addr - sistrip::APV_I2C_MIN ) % 2 ) == 0 );
}

// -----------------------------------------------------------------------------
// 
bool SiStripFecKey::isEqual( const SiStripKey& key ) const {
  const SiStripFecKey& input = dynamic_cast<const SiStripFecKey&>(key);
  if ( !(&input) ) { return false; }
  if ( fecCrate_ == input.fecCrate() &&
       fecSlot_ == input.fecSlot() &&
       fecRing_ == input.fecRing() &&
       ccuAddr_ == input.ccuAddr() &&
       ccuChan_ == input.ccuChan() &&
       lldChan_ == input.lldChan() &&
       i2cAddr_ == input.i2cAddr() ) { 
    return true;
  } else { return false; }
}

// -----------------------------------------------------------------------------
// 
bool SiStripFecKey::isConsistent( const SiStripKey& key ) const {
  const SiStripFecKey& input = dynamic_cast<const SiStripFecKey&>(key);
  if ( !(&input) ) { return false; }
  if ( isEqual(input) ) { return true; }
  else if ( ( fecCrate_ == 0 || input.fecCrate() == 0 ) &&
	    ( fecSlot_ == 0 || input.fecSlot() == 0 ) &&
	    ( fecRing_ == 0 || input.fecRing() == 0 ) &&
	    ( ccuAddr_ == 0 || input.ccuAddr() == 0 ) &&
	    ( lldChan_ == 0 || input.lldChan() == 0 ) &&
	    ( i2cAddr_ == 0 || input.i2cAddr() == 0 ) ) {
    return true;
  } else { return false; }
}

// -----------------------------------------------------------------------------
//
bool SiStripFecKey::isValid() const { 
  return isValid(sistrip::APV); 
}

// -----------------------------------------------------------------------------
//
bool SiStripFecKey::isValid( const sistrip::Granularity& gran ) const {
  if ( gran == sistrip::FEC_SYSTEM ) { return true; }
  else if ( gran == sistrip::UNDEFINED_GRAN ||
	    gran == sistrip::UNKNOWN_GRAN ) { return false; }
  
  if ( fecCrate_ != sistrip::invalid_ ) {
    if ( gran == sistrip::FEC_CRATE ) { return true; }
    if ( fecSlot_ != sistrip::invalid_ ) {
      if ( gran == sistrip::FEC_RING ) { return true; }
      if ( fecRing_ != sistrip::invalid_ ) {
	if ( gran == sistrip::FEC_RING ) { return true; }
	if ( ccuAddr_ != sistrip::invalid_ ) {
	  if ( gran == sistrip::CCU_ADDR ) { return true; }
	  if ( ccuChan_ != sistrip::invalid_ ) {
	    if ( gran == sistrip::CCU_CHAN ) { return true; }
	    if ( lldChan_ != sistrip::invalid_ ) {
	      if ( gran == sistrip::LLD_CHAN ) { return true; }
	      if ( i2cAddr_ != sistrip::invalid_ ) {
		if ( gran == sistrip::APV ) { return true; }
	      }
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
bool SiStripFecKey::isInvalid() const { 
  return isInvalid(sistrip::APV); 
}

// -----------------------------------------------------------------------------
//
bool SiStripFecKey::isInvalid( const sistrip::Granularity& gran ) const {
  if ( gran == sistrip::FEC_SYSTEM ) { return false; }
  else if ( gran == sistrip::UNDEFINED_GRAN ||
	    gran == sistrip::UNKNOWN_GRAN ) { return false; }
  
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
	    if ( lldChan_ == sistrip::invalid_ ) {
	      if ( gran == sistrip::LLD_CHAN  ) { return true; }
	      if ( i2cAddr_ == sistrip::invalid_ ) {
		if ( gran == sistrip::APV  ) { return true; }
	      }
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
void SiStripFecKey::initFromValue() {

  // FEC crate  
  if ( fecCrate_ >= sistrip::FEC_CRATE_MIN &&
       fecCrate_ <= sistrip::FEC_CRATE_MAX ) {
    fecCrate_ = fecCrate_;
  } else if ( fecCrate_ == 0 ) { 
    fecCrate_ = 0;
  } else { fecCrate_ = sistrip::invalid_; }

  // FEC slot
  if ( fecSlot_ >= sistrip::CRATE_SLOT_MIN &&
       fecSlot_ <= sistrip::CRATE_SLOT_MAX ) {
    fecSlot_ = fecSlot_;
  } else if ( fecSlot_ == 0 ) { 
    fecSlot_ = 0;
  } else { fecSlot_ = sistrip::invalid_; }

  // FEC ring
  if ( fecRing_ >= sistrip::FEC_RING_MIN &&
       fecRing_ <= sistrip::FEC_RING_MAX ) {
    fecRing_ = fecRing_;
  } else if ( fecRing_ == 0 ) { 
    fecRing_ = 0;
  } else { fecRing_ = sistrip::invalid_; }

  // CCU addr
  if ( ccuAddr_ >= sistrip::CCU_ADDR_MIN &&
       ccuAddr_ <= sistrip::CCU_ADDR_MAX ) {
    ccuAddr_ = ccuAddr_;
  } else if ( ccuAddr_ == 0 ) { 
    ccuAddr_ = 0;
  } else { ccuAddr_ = sistrip::invalid_; }

  // CCU chan
  if ( ccuChan_ >= sistrip::CCU_CHAN_MIN &&
       ccuChan_ <= sistrip::CCU_CHAN_MAX ) {
    ccuChan_ = ccuChan_;
  } else if ( ccuChan_ == 0 ) { 
    ccuChan_ = 0;
  } else { ccuChan_ = sistrip::invalid_; }
  
  // LLD channel
  if ( lldChan_ >= sistrip::LLD_CHAN_MIN &&
       lldChan_ <= sistrip::LLD_CHAN_MAX ) {
    lldChan_ = lldChan_;
  } else if ( lldChan_ == 0 ) { 
    lldChan_ = 0;
  } else { lldChan_ = sistrip::invalid_; }
  
  // APV I2C address
  if ( i2cAddr_ >= sistrip::APV_I2C_MIN &&
       i2cAddr_ <= sistrip::APV_I2C_MAX ) { 
    i2cAddr_ = i2cAddr_; 
    if ( lldChan_ && lldChan( i2cAddr_ ) != lldChan_ ) { 
      i2cAddr_ = sistrip::invalid_;
      key( key() | (i2cAddrMask_<<i2cAddrOffset_) ); 
    }
  } else if ( i2cAddr_ == 0 ) { 
    i2cAddr_ = 0;
  } else { i2cAddr_ = sistrip::invalid_; }
  
}

// -----------------------------------------------------------------------------
//
void SiStripFecKey::initFromKey() {
  
  if ( key() == sistrip::invalid32_ ) { 

    // ---------- Set FecKey based on member data ----------
    
    // Initialise to null value
    key(0);
    
    // Extract FEC crate  
    if ( fecCrate_ >= sistrip::FEC_CRATE_MIN &&
	 fecCrate_ <= sistrip::FEC_CRATE_MAX ) {
      key( key() | (fecCrate_<<fecCrateOffset_) );
    } else if ( fecCrate_ == 0 ) { 
      key( key() | (fecCrate_<<fecCrateOffset_) );
    } else { 
      key( key() | (fecCrateMask_<<fecCrateOffset_) ); 
    }

    // Extract FEC slot
    if ( fecSlot_ >= sistrip::CRATE_SLOT_MIN &&
	 fecSlot_ <= sistrip::CRATE_SLOT_MAX ) {
      key( key() | (fecSlot_<<fecSlotOffset_) );
    } else if ( fecSlot_ == 0 ) { 
      key( key() | (fecSlot_<<fecSlotOffset_) );
    } else { 
      key( key() | (fecSlotMask_<<fecSlotOffset_) ); 
    }

    // Extract FEC ring
    if ( fecRing_ >= sistrip::FEC_RING_MIN &&
	 fecRing_ <= sistrip::FEC_RING_MAX ) {
      key( key() | (fecRing_<<fecRingOffset_) );
    } else if ( fecRing_ == 0 ) { 
      key( key() | (fecRing_<<fecRingOffset_) );
    } else { 
      key( key() | (fecRingMask_<<fecRingOffset_) ); 
    }

    // Extract CCU addr
    if ( ccuAddr_ >= sistrip::CCU_ADDR_MIN &&
	 ccuAddr_ <= sistrip::CCU_ADDR_MAX ) {
      key( key() | (ccuAddr_<<ccuAddrOffset_) );
    } else if ( ccuAddr_ == 0 ) { 
      key( key() | (ccuAddr_<<ccuAddrOffset_) );
    } else { 
      key( key() | (ccuAddrMask_<<ccuAddrOffset_) ); 
    }

    // Extract CCU chan
    if ( ccuChan_ >= sistrip::CCU_CHAN_MIN &&
	 ccuChan_ <= sistrip::CCU_CHAN_MAX ) {
      key( key() | ( (ccuChan_-(sistrip::CCU_CHAN_MIN-1)) << ccuChanOffset_ ) ); 
    } else if ( ccuChan_ == 0 ) { 
      key( key() | (ccuChan_<<ccuChanOffset_) );
    } else { 
      key( key() | (ccuChanMask_<<ccuChanOffset_) ); 
    }
    
    // Extract LLD channel
    if ( lldChan_ >= sistrip::LLD_CHAN_MIN &&
	 lldChan_ <= sistrip::LLD_CHAN_MAX ) {
      key( key() | (lldChan_<<lldChanOffset_) ); 
    } else if ( lldChan_ == 0 ) { 
      key( key() | (lldChan_<<lldChanOffset_) );
    } else { 
      key( key() | (lldChanMask_<<lldChanOffset_) ); 
    }
    
    // Extract APV I2C address
    if ( i2cAddr_ >= sistrip::APV_I2C_MIN &&
	 i2cAddr_ <= sistrip::APV_I2C_MAX ) {
      key( key() | ( ( firstApvOfPair( i2cAddr_ ) ? 1 : 2 ) << i2cAddrOffset_ ) ); // key encodes APV number (1 or 2)
      if ( lldChan_ && lldChan( i2cAddr_ ) != lldChan_ ) { 
	i2cAddr_ = sistrip::invalid_;
	key( key() | (i2cAddrMask_<<i2cAddrOffset_) ); 
      }
    } else if ( i2cAddr_ == 0 ) { 
      key( key() | (i2cAddr_<<i2cAddrOffset_) );
    } else { 
      key( key() | (i2cAddrMask_<<i2cAddrOffset_) ); 
    }
    
  } else {
    
    // ---------- Set member data based on FEC key ----------

    fecCrate_ = ( key()>>fecCrateOffset_ ) & fecCrateMask_;
    fecSlot_  = ( key()>>fecSlotOffset_ )  & fecSlotMask_;
    fecRing_  = ( key()>>fecRingOffset_ )  & fecRingMask_;
    ccuAddr_  = ( key()>>ccuAddrOffset_ )  & ccuAddrMask_;
    ccuChan_  = ( key()>>ccuChanOffset_ )  & ccuChanMask_;
    lldChan_  = ( key()>>lldChanOffset_ )  & lldChanMask_;
    i2cAddr_  = ( key()>>i2cAddrOffset_ )  & i2cAddrMask_;

    if ( fecCrate_ == fecCrateMask_ ) { fecCrate_ = sistrip::invalid_; } 
    if ( fecSlot_ == fecSlotMask_ ) { fecSlot_ = sistrip::invalid_; } 
    if ( fecRing_ == fecRingMask_ ) { fecRing_ = sistrip::invalid_; } 
    if ( ccuAddr_ == ccuAddrMask_ ) { ccuAddr_ = sistrip::invalid_; } 
    if ( ccuChan_ == ccuChanMask_ ) { ccuChan_ = sistrip::invalid_; }
    else if ( ccuChan_ ) { ccuChan_ += (sistrip::CCU_CHAN_MIN-1); }
    if ( lldChan_ == lldChanMask_ ) { lldChan_ = sistrip::invalid_; }
    if ( i2cAddr_ == i2cAddrMask_ ) { i2cAddr_ = sistrip::invalid_; }
    else if ( i2cAddr_ && lldChan_ != lldChanMask_ ) { i2cAddr_ = i2cAddr( lldChan_, 2-i2cAddr_ ); }
    
  }
  
}

// -----------------------------------------------------------------------------
// 
void SiStripFecKey::initFromPath() {
  
  if ( path() == sistrip::null_ ) {
    
    // ---------- Set directory path based on member data ----------

    std::stringstream dir;
    
    dir << sistrip::root_ << sistrip::dir_ 
	<< sistrip::controlView_ << sistrip::dir_;

    // Add FEC crate
    if ( fecCrate_ ) {
      dir << sistrip::fecCrate_ << fecCrate_ << sistrip::dir_;
      
      // Add FEC slot
      if ( fecSlot_ ) {
	dir << sistrip::fecSlot_ << fecSlot_ << sistrip::dir_;
	
	// Add FEC ring
	if ( fecRing_ ) {
	  dir << sistrip::fecRing_ << fecRing_ << sistrip::dir_;
	  
	  // Add CCU address
	  if ( ccuAddr_ ) {
	    dir << sistrip::ccuAddr_ << ccuAddr_ << sistrip::dir_;
	    
	    // Add CCU channel
	    if ( ccuChan_ ) {
	      dir << sistrip::ccuChan_ << ccuChan_ << sistrip::dir_;

	      // Add LLD channel
	      if ( lldChan_ ) {
		dir << sistrip::lldChan_ << lldChan_ << sistrip::dir_;

		// Add APV I2C address
		if ( i2cAddr_ ) {
		  dir << sistrip::apv_ << i2cAddr_ << sistrip::dir_;
		}
	      }
	    }
	  }
	}
      }
    }
    
    std::string temp( dir.str() );
    path( temp );

  } else {
    
    // ---------- Set member data based on directory path ----------
    
    fecCrate_ = 0;
    fecSlot_  = 0;
    fecRing_  = 0;
    ccuAddr_  = 0;
    ccuChan_  = 0;
    lldChan_  = 0;
    i2cAddr_  = 0;

    // Check if root is found
    if ( path().find( sistrip::root_ ) == std::string::npos ) {
      std::string temp = path();
      path( std::string(sistrip::root_) + sistrip::dir_ + temp );
    }
    
    size_t curr = 0; // current string position
    size_t next = 0; // next string position
    next = path().find( sistrip::controlView_, curr );

    // Extract view 
    curr = next;
    if ( curr != std::string::npos ) { 
      next = path().find( sistrip::fecCrate_, curr );
      std::string control_view( path(), 
				curr+(sizeof(sistrip::controlView_) - 1), 
				next-(sizeof(sistrip::dir_) - 1)-curr );
      
      // Extract FEC crate
      curr = next;
      if ( curr != std::string::npos ) { 
	next = path().find( sistrip::fecSlot_, curr );
	std::string fec_crate( path(), 
			       curr+(sizeof(sistrip::fecCrate_) - 1), 
			       next-(sizeof(sistrip::dir_) - 1)-curr );
	fecCrate_ = std::atoi( fec_crate.c_str() );

	// Extract FEC slot
	curr = next;
	if ( curr != std::string::npos ) { 
	  next = path().find( sistrip::fecRing_, curr );
	  std::string fec_slot( path(), 
				curr+(sizeof(sistrip::fecSlot_) - 1), 
				next-(sizeof(sistrip::dir_) - 1)-curr );
	  fecSlot_ = std::atoi( fec_slot.c_str() );

	  // Extract FEC ring
	  curr = next;
	  if ( curr != std::string::npos ) { 
	    next = path().find( sistrip::ccuAddr_, curr );
	    std::string fec_ring( path(), 
				  curr+(sizeof(sistrip::fecRing_) - 1),
				  next-(sizeof(sistrip::dir_) - 1)-curr );
	    fecRing_ = std::atoi( fec_ring.c_str() );

	    // Extract CCU address
	    curr = next;
	    if ( curr != std::string::npos ) { 
	      next = path().find( sistrip::ccuChan_, curr );
	      std::string ccu_addr( path(), 
				    curr+(sizeof(sistrip::ccuAddr_) - 1), 
				    next-(sizeof(sistrip::dir_) - 1)-curr );
	      ccuAddr_ = std::atoi( ccu_addr.c_str() );

	      // Extract CCU channel
	      curr = next;
	      if ( curr != std::string::npos ) { 
		next = path().find( sistrip::lldChan_, curr );
		std::string ccu_chan( path(), 
				      curr+(sizeof(sistrip::ccuChan_) - 1), 
				      next-(sizeof(sistrip::dir_) - 1)-curr );
		ccuChan_ = std::atoi( ccu_chan.c_str() );
		
		// Extract LLD channel
		curr = next;
		if ( curr != std::string::npos ) { 
		  next = path().find( sistrip::apv_, curr );
		  std::string lld_chan( path(), 
					curr+(sizeof(sistrip::lldChan_) - 1), 
					next-(sizeof(sistrip::dir_) - 1)-curr );
		  lldChan_ = std::atoi( lld_chan.c_str() );
		  
		  // Extract I2C address
		  curr = next;
		  if ( curr != std::string::npos ) { 
		    next = std::string::npos;
		    std::string i2c_addr( path(), 
					  curr+(sizeof(sistrip::apv_) - 1),
					  next-curr );
		    i2cAddr_ = std::atoi( i2c_addr.c_str() );
		  }
		}
	      }
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
void SiStripFecKey::initGranularity() {
  
  granularity( sistrip::FEC_SYSTEM );
  channel(0);
  if ( fecCrate_ && fecCrate_ != sistrip::invalid_ ) {
    granularity( sistrip::FEC_CRATE ); 
    channel(fecCrate_);
    if ( fecSlot_ && fecSlot_ != sistrip::invalid_ ) {
      granularity( sistrip::FEC_SLOT );
      channel(fecSlot_);
      if ( fecRing_ && fecRing_ != sistrip::invalid_ ) {
	granularity( sistrip::FEC_RING );
	channel(fecRing_);
	if ( ccuAddr_ && ccuAddr_ != sistrip::invalid_ ) {
	  granularity( sistrip::CCU_ADDR );
	  channel(ccuAddr_);
	  if ( ccuChan_ && ccuChan_ != sistrip::invalid_ ) {
	    granularity( sistrip::CCU_CHAN );
	    channel(ccuChan_);
	    if ( lldChan_ && lldChan_ != sistrip::invalid_ ) {
	      granularity( sistrip::LLD_CHAN );
	      channel(lldChan_);
	      if ( i2cAddr_ && i2cAddr_ != sistrip::invalid_ ) {
		granularity( sistrip::APV );
		channel(i2cAddr_);
	      } else if ( i2cAddr_ == sistrip::invalid_ ) { 
		granularity( sistrip::UNKNOWN_GRAN ); 
		channel(sistrip::invalid_);
	      }
	    } else if ( lldChan_ == sistrip::invalid_ ) { 
	      granularity( sistrip::UNKNOWN_GRAN ); 
	      channel(sistrip::invalid_);
	    }
	  } else if ( ccuChan_ == sistrip::invalid_ ) { 
	    granularity( sistrip::UNKNOWN_GRAN ); 
	    channel(sistrip::invalid_);
	  }
	} else if ( ccuAddr_ == sistrip::invalid_ ) { 
	  granularity( sistrip::UNKNOWN_GRAN ); 
	  channel(sistrip::invalid_);
	}
      } else if ( fecRing_ == sistrip::invalid_ ) { 
	granularity( sistrip::UNKNOWN_GRAN ); 
	channel(sistrip::invalid_);
      }
    } else if ( fecSlot_ == sistrip::invalid_ ) { 
      granularity( sistrip::UNKNOWN_GRAN ); 
      channel(sistrip::invalid_);
    }
  } else if ( fecCrate_ == sistrip::invalid_ ) { 
    granularity( sistrip::UNKNOWN_GRAN ); 
    channel(sistrip::invalid_);
  }

}

// -----------------------------------------------------------------------------
//
void SiStripFecKey::terse( std::stringstream& ss ) const {
  ss << "FEC:crate/slot/ring/CCU/module/LLD/I2C= "
     << fecCrate() << "/"
     << fecSlot() << "/"
     << fecRing() << "/"
     << ccuAddr() << "/"
     << ccuChan() << "/"
     << lldChan() << "/"
     << i2cAddr();
//   ss << " FecKey"
//     //<< "=0x" 
//     //<< std::hex
//     //<< std::setfill('0') << std::setw(8) << key() << std::setfill(' ') 
//     //<< std::dec
//     //<< ", " << ( isValid() ? "Valid" : "Invalid" )
//      << ", Crate=" << fecCrate()
//      << ", Slot=" << fecSlot()
//      << ", Ring=" << fecRing()
//      << ", CCU=" << ccuAddr()
//      << ", module=" << ccuChan()
//      << ", LLD=" << lldChan()
//      << ", I2C=" << i2cAddr();
}

// -----------------------------------------------------------------------------
//
void SiStripFecKey::print( std::stringstream& ss ) const {
  ss << " [SiStripFecKey::print]" << std::endl
     << std::hex
     << " FEC key              : 0x" 
     << std::setfill('0') 
     << std::setw(8) << key() << std::endl
     << std::setfill(' ') 
     << std::dec
     << " FEC VME crate        : " << fecCrate() << std::endl
     << " FEC VME slot         : " << fecSlot() << std::endl 
     << " FEC control ring     : " << fecRing() << std::endl
     << " CCU I2C address      : " << ccuAddr() << std::endl
     << " CCU chan (FE module) : " << ccuChan() << std::endl
     << " LaserDriver channel  : " << lldChan() << std::endl 
     << " APV I2C address      : " << i2cAddr() << std::endl 
     << " Directory            : " << path() << std::endl
     << " Granularity          : "
     << SiStripEnumsAndStrings::granularity( granularity() ) << std::endl
     << " Channel              : " << channel() << std::endl
     << " isValid              : " << isValid();
}

// -----------------------------------------------------------------------------
//
std::ostream& operator<< ( std::ostream& os, const SiStripFecKey& input ) {
  std::stringstream ss;
  input.print(ss);
  os << ss.str();
  return os;
}

// -----------------------------------------------------------------------------
//
ConsistentWithKey::ConsistentWithKey( const SiStripFecKey& key ) 
  : mask_( key.fecCrate() ? sistrip::invalid_ : 0,
 	   key.fecSlot() ? sistrip::invalid_ : 0,
 	   key.fecRing() ? sistrip::invalid_ : 0,
 	   key.ccuAddr() ? sistrip::invalid_ : 0,
 	   key.ccuChan() ? sistrip::invalid_ : 0,
 	   key.lldChan() ? sistrip::invalid_ : 0,
 	   key.i2cAddr() ? sistrip::invalid_ : 0 ) {;}

// -----------------------------------------------------------------------------
//
ConsistentWithKey::ConsistentWithKey() 
  : mask_(SiStripNullKey()) {;}

// -----------------------------------------------------------------------------
//
bool ConsistentWithKey::operator() ( const uint32_t& a, const uint32_t& b ) const {
  return ( ( a & mask_.key() ) < ( b & mask_.key() ) );
}
