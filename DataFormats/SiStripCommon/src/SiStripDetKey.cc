// Last commit: $Id: SiStripDetKey.cc,v 1.12 2012/07/04 19:04:49 eulisse Exp $

#include "DataFormats/SiStripCommon/interface/SiStripDetKey.h"
#include "DataFormats/SiStripCommon/interface/Constants.h" 
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include <iomanip>

// -----------------------------------------------------------------------------
// 
SiStripDetKey::SiStripDetKey( const uint16_t& partition ) :
  SiStripKey(),
  partition_(partition),
  apvPairNumber_(sistrip::invalid_), 
  apvWithinPair_(sistrip::invalid_)
{
  // order is important!
  initFromValue();
  initFromKey();
  initFromPath();
  initGranularity();
}

// -----------------------------------------------------------------------------
// 
SiStripDetKey::SiStripDetKey( const DetId& det_id,
			      const uint16_t& apv_pair_number,
			      const uint16_t& apv_within_pair ) :
  SiStripKey(),
  partition_(sistrip::invalid_),
  apvPairNumber_(apv_pair_number), 
  apvWithinPair_(apv_within_pair)
{
  // order is important!
  initFromValue();
  initFromKey();
  initFromPath();
  initGranularity();
}

// -----------------------------------------------------------------------------
// 
SiStripDetKey::SiStripDetKey( const SiStripDetId& det_id ) :
  SiStripKey(),
  partition_(sistrip::invalid_),
  apvPairNumber_(sistrip::invalid_), 
  apvWithinPair_(sistrip::invalid_)
{
  // order is important!
  initFromValue();
  initFromKey();
  initFromPath();
  initGranularity();
}

// -----------------------------------------------------------------------------
// 
SiStripDetKey::SiStripDetKey( const uint32_t& det_key ) :
  SiStripKey(det_key),
  partition_(sistrip::invalid_),
  apvPairNumber_(sistrip::invalid_), 
  apvWithinPair_(sistrip::invalid_)
{
  // order is important!
  initFromKey(); 
  initFromValue();
  initFromPath();
  initGranularity();
}

// -----------------------------------------------------------------------------
// 
SiStripDetKey::SiStripDetKey( const std::string& path ) :
  SiStripKey(path),
  partition_(sistrip::invalid_),
  apvPairNumber_(sistrip::invalid_), 
  apvWithinPair_(sistrip::invalid_)
{
  // order is important!
  initFromPath();
  initFromValue();
  initFromKey();
  initGranularity();
}

// -----------------------------------------------------------------------------
// 
SiStripDetKey::SiStripDetKey( const SiStripDetKey& input ) :
  SiStripKey(),
  partition_(input.partition()),
  apvPairNumber_(input.apvPairNumber()), 
  apvWithinPair_(input.apvWithinPair())
{
  key(input.key());
  path(input.path());
  granularity(input.granularity());
}

// -----------------------------------------------------------------------------
// 
SiStripDetKey::SiStripDetKey( const SiStripKey& input ) :
  SiStripKey(),
  partition_(sistrip::invalid_),
  apvPairNumber_(sistrip::invalid_), 
  apvWithinPair_(sistrip::invalid_)
{
  const SiStripDetKey& det_key = dynamic_cast<const SiStripDetKey&>(input);
  if ( (&det_key) ) {
    key(det_key.key());
    path(det_key.path());
    granularity(det_key.granularity());
    partition_ = det_key.partition();
    apvPairNumber_ = det_key.apvPairNumber();
    apvWithinPair_ = det_key.apvWithinPair();
  }
}

// -----------------------------------------------------------------------------
// 
SiStripDetKey::SiStripDetKey( const SiStripKey& input,
			      const sistrip::Granularity& gran ) :
  SiStripKey(),
  partition_(0),
  apvPairNumber_(0),
  apvWithinPair_(0)
{
  const SiStripDetKey& det_key = dynamic_cast<const SiStripDetKey&>(input);
  if ( (&det_key) ) {

    if ( gran == sistrip::PARTITION ) {
      partition_ = det_key.partition(); 
    }

    initFromValue();
    initFromKey();
    initFromPath();
    initGranularity();
    
  }

}

// -----------------------------------------------------------------------------
// 
SiStripDetKey::SiStripDetKey() : 
  SiStripKey(),
  partition_(sistrip::invalid_),
  apvPairNumber_(sistrip::invalid_), 
  apvWithinPair_(sistrip::invalid_)
{;}

// -----------------------------------------------------------------------------
// 
bool SiStripDetKey::isEqual( const SiStripKey& key ) const {
  const SiStripDetKey& input = dynamic_cast<const SiStripDetKey&>(key);
  if ( !(&input) ) { return false; }
  if ( partition_ == input.partition() &&
       apvPairNumber_ == input.apvPairNumber() &&
       apvWithinPair_ == input.apvWithinPair() ) {
    return true;
  } else { return false; }
}

// -----------------------------------------------------------------------------
// 
bool SiStripDetKey::isConsistent( const SiStripKey& key ) const {
  const SiStripDetKey& input = dynamic_cast<const SiStripDetKey&>(key);
  if ( !(&input) ) { return false; }
  if ( isEqual(input) ) { return false; }
  else if ( ( partition_ == 0 || input.partition() == 0 ) &&
            ( apvPairNumber_ == 0 || input.apvPairNumber() == 0 ) &&
            ( apvWithinPair_ == 0 || input.apvWithinPair() == 0 ) ) {
    return true;
  } else { return false; }
}

// -----------------------------------------------------------------------------
//
bool SiStripDetKey::isValid() const { 
  return isValid(sistrip::APV); 
}

// -----------------------------------------------------------------------------
//
bool SiStripDetKey::isValid( const sistrip::Granularity& gran ) const {
  if ( gran == sistrip::TRACKER ) { return true; }
  else if ( gran == sistrip::UNDEFINED_GRAN ||
	    gran == sistrip::UNKNOWN_GRAN ) { return false; }

  if ( partition_ != sistrip::invalid_ ) {
    if ( gran == sistrip::PARTITION ) { return true; }
  }
  return false;
}

// -----------------------------------------------------------------------------
//
bool SiStripDetKey::isInvalid() const { 
  return isInvalid(sistrip::APV); 
}

// -----------------------------------------------------------------------------
//
bool SiStripDetKey::isInvalid( const sistrip::Granularity& gran ) const {
  if ( gran == sistrip::TRACKER ) { return false; }
  else if ( gran == sistrip::UNDEFINED_GRAN ||
	    gran == sistrip::UNKNOWN_GRAN ) { return false; }

  if ( partition_ == sistrip::invalid_ ) {
    if ( gran == sistrip::PARTITION ) { return true; }
  }
  return false;
}

// -----------------------------------------------------------------------------
// 
void SiStripDetKey::initFromValue() {

  // partition
  if ( partition_ >= 1 && //sistrip::PARTITION_MIN &&
       partition_ <= 4 ) { //sistrip::PARTITION_MAX ) {
    partition_ = partition_;
  } else if ( partition_ == 0 ) {
    partition_ = 0;
  } else { partition_ = sistrip::invalid_; }

}

// -----------------------------------------------------------------------------
//
void SiStripDetKey::initFromKey() {

  if ( key() == sistrip::invalid32_ ) {

    // ---------- Set DetKey based on member data ----------
    
    // Initialise to null value
    key(0);
    
    // Extract partition
    if ( partition_ >= 1 && //sistrip::PARTITION_MIN &&
         partition_ <= 4 ) { //sistrip::PARTITION_MAX ) {
      key( key() | (partition_<<partitionOffset_) );
    } else if ( partition_ == 0 ) {
      key( key() | (partition_<<partitionOffset_) );
    } else {
      key( key() | (partitionMask_<<partitionOffset_) );
    }
    
  } else {
    
    // ---------- Set member data based on Det key ----------

    partition_ = ( key()>>partitionOffset_ ) & partitionMask_;

    if ( partition_ == partitionMask_ ) { partition_ = sistrip::invalid_; }

  }

}

// -----------------------------------------------------------------------------
// 
void SiStripDetKey::initFromPath() {

  if ( path() == sistrip::null_ ) {

    // ---------- Set directory path based on member data ----------

    std::stringstream dir;

    dir << sistrip::root_ << sistrip::dir_ 
	<< sistrip::detectorView_ << sistrip::dir_;

    // Add partition
    if ( partition_ ) {
      dir << sistrip::partition_ << partition_ << sistrip::dir_;
    }

    std::string temp( dir.str() );
    path( temp );

  } else {

    // ---------- Set member data based on directory path ----------

    partition_ = 0;

    // Check if root is found
    if ( path().find( sistrip::root_ ) == std::string::npos ) {
      std::string temp = path();
      path( std::string(sistrip::root_) + sistrip::dir_ + temp );
    }

    size_t curr = 0; // current string position
    size_t next = 0; // next string position
    next = path().find( sistrip::detectorView_, curr );

    // Extract view
    curr = next;
    if ( curr != std::string::npos ) {
      next = path().find( sistrip::partition_, curr );
      std::string detector_view( path(),
				 curr+(sizeof(sistrip::detectorView_) - 1),
				 next-(sizeof(sistrip::dir_) - 1)-curr );
      // Extract partition
      curr = next;
      if ( curr != std::string::npos ) { 
        next = std::string::npos;
        std::string partition( path(), 
                               curr+(sizeof(sistrip::partition_) - 1),
                               next-(sizeof(sistrip::dir_) - 1)-curr );
        partition_ = std::atoi( partition.c_str() );
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
void SiStripDetKey::initGranularity() {

  granularity( sistrip::TRACKER );
  channel(0);
  if ( partition_ && partition_ != sistrip::invalid_ ) {
    granularity( sistrip::PARTITION );
    channel(partition_);
  } else if ( partition_ == sistrip::invalid_ ) { 
    granularity( sistrip::UNKNOWN_GRAN );
    channel(sistrip::invalid_);
  }

}

// -----------------------------------------------------------------------------
//
void SiStripDetKey::terse( std::stringstream& ss ) const {
  ss << "DET:partition= "
     << partition();
}

// -----------------------------------------------------------------------------
//
void SiStripDetKey::print( std::stringstream& ss ) const {
  ss << " [SiStripDetKey::print]" << std::endl
     << std::hex
     << " 32-bit Det key : 0x"
     << std::setfill('0') 
     << std::setw(8) << key() << std::endl
     << std::setfill(' ')
     << std::dec
     << " Partition      : " << partition() << std::endl
     << " Directory      : " << path() << std::endl
     << " Granularity    : "
     << SiStripEnumsAndStrings::granularity( granularity() ) << std::endl
     << " Channel        : " << channel() << std::endl
     << " isValid        : " << isValid();
}

// -----------------------------------------------------------------------------
//
std::ostream& operator<< ( std::ostream& os, const SiStripDetKey& input ) {
  std::stringstream ss;
  input.print(ss);
  os << ss.str();
  return os;
}
