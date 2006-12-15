#include "DataFormats/SiStripCommon/interface/SiStripDetKey.h"
#include <iomanip>

// -----------------------------------------------------------------------------
// 
SiStripDetKey::Path::Path() 
  : detId_(sistrip::invalid_), 
    apvPair_(sistrip::invalid_) {;}

// -----------------------------------------------------------------------------
// 
SiStripDetKey::Path::Path( const uint32_t& det_id,
	    const uint16_t& apv_pair ) 
  : detId_(det_id), 
    apvPair_(apv_pair) {;}

// -----------------------------------------------------------------------------
// 
bool SiStripDetKey::Path::isEqual( const Path& input ) {
  if ( detId_ == input.detId_ &&
       apvPair_ == input.apvPair_ ) { 
    return true;
  } else { return false; }
}

// -----------------------------------------------------------------------------
// 
bool SiStripDetKey::Path::isConsistent( const Path& input ) {
  if ( isEqual(input) ) { return true; }
  else if ( ( detId_ == 0 || input.detId_ == 0 ) &&
	    ( apvPair_ == 0 || input.apvPair_ == 0 ) ) { 
    return true;
  } else { return false; }
}

// -----------------------------------------------------------------------------
//
bool SiStripDetKey::Path::isInvalid() const {
  if ( detId_ == sistrip::invalid_ &&
       apvPair_ == sistrip::invalid_ ) {
    return true;
  } else { return false; }
}

// -----------------------------------------------------------------------------
//
bool SiStripDetKey::Path::isInvalid( const sistrip::Granularity& gran ) const {
  if ( detId_ == sistrip::invalid_ ) {
    if ( gran == sistrip::MODULE ) { return true; }
    if ( apvPair_ == sistrip::invalid_ ) {
      if ( gran == sistrip::LLD_CHAN ) { return true; }
    }
  }
  return false;
}

// -----------------------------------------------------------------------------
// 
uint32_t SiStripDetKey::key( const uint32_t& det_id,
			     const uint16_t& apv_pair ) { 
  return static_cast<uint32_t>( SiStripDetId(det_id,apv_pair).rawId() );
}

// -----------------------------------------------------------------------------
// 
uint32_t SiStripDetKey::key( const DetId& det_id,
			     const uint16_t& apv_pair ) { 
  return static_cast<uint32_t>( SiStripDetId(det_id.rawId(),apv_pair).rawId() );
}

// -----------------------------------------------------------------------------
// 
uint32_t SiStripDetKey::key( const Path& path ) {
  return SiStripDetKey::key( path.detId_,
			     path.apvPair_ );
}

// -----------------------------------------------------------------------------
// 
SiStripDetKey::Path SiStripDetKey::path( const uint32_t& det_id ) { 
  return Path( static_cast<uint32_t>( SiStripDetId(det_id,0).rawId() ), 
	       static_cast<uint16_t>( SiStripDetId(det_id).reserved() ) );
}

// -----------------------------------------------------------------------------
// 
SiStripDetKey::Path SiStripDetKey::path( const SiStripDetId& det_id ) { 
  return Path( static_cast<uint32_t>( SiStripDetId(det_id,0).rawId() ), 
	       static_cast<uint16_t>( SiStripDetId(det_id).reserved() ) );
}

// -----------------------------------------------------------------------------
//
bool SiStripDetKey::isEqual( const uint32_t& key1, 
			     const uint32_t& key2 ) {
  SiStripDetKey::Path path1 = SiStripDetKey::path( key1 ) ;
  SiStripDetKey::Path path2 = SiStripDetKey::path( key2 ) ;
  return path1.isEqual( path2 );
}

// -----------------------------------------------------------------------------
//
bool SiStripDetKey::isConsistent( const uint32_t& key1, 
				  const uint32_t& key2 ) {
  SiStripDetKey::Path path1 = SiStripDetKey::path( key1 ) ;
  SiStripDetKey::Path path2 = SiStripDetKey::path( key2 ) ;
  return path1.isConsistent( path2 );
}

// -----------------------------------------------------------------------------
//
std::ostream& operator<< ( std::ostream& os, const SiStripDetKey::Path& path ) {
  return os << "[SiStripDetKey::Path]"
	    << std::hex
	    << " key=0x" << std::setfill('0') << std::setw(8) << SiStripDetId(path.detId_,path.apvPair_).rawId()
	    << " detId=0x" << std::setfill('0') << std::setw(8) << path.detId_ 
	    << std::dec
	    << " apvPair=" << path.apvPair_;
}
