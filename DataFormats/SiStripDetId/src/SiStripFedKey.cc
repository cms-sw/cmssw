#include "DataFormats/SiStripDetId/interface/SiStripFedKey.h"
#include <iomanip>

// -----------------------------------------------------------------------------
// 
uint32_t SiStripFedKey::key( uint16_t fed_id,
			     uint16_t fed_fe,
			     uint16_t fed_ch ) {
  return( static_cast<uint32_t>( ( fed_id & SiStripFedKey::fedIdMask_ ) << SiStripFedKey::fedIdOffset_ ) | 
	  static_cast<uint32_t>( ( fed_fe & SiStripFedKey::fedFeMask_ ) << SiStripFedKey::fedFeOffset_ ) | 
	  static_cast<uint32_t>( ( fed_ch & SiStripFedKey::fedChMask_ ) << SiStripFedKey::fedChOffset_ ) );
}

// -----------------------------------------------------------------------------
// 
uint32_t SiStripFedKey::key( const Path& path ) {
  return SiStripFedKey::key( path.fedId_,
			     path.fedFe_,
			     path.fedCh_ );
}

// -----------------------------------------------------------------------------
//
SiStripFedKey::Path SiStripFedKey::path( uint32_t key ) {
  Path tmp;
  tmp.fedId_ = ( key>>SiStripFedKey::fedIdOffset_ ) & SiStripFedKey::fedIdMask_;
  tmp.fedFe_ = ( key>>SiStripFedKey::fedFeOffset_ ) & SiStripFedKey::fedFeMask_;
  tmp.fedCh_ = ( key>>SiStripFedKey::fedChOffset_ ) & SiStripFedKey::fedChMask_;
  if ( tmp.fedId_ == SiStripFedKey::fedIdMask_ ) { tmp.fedId_ = sistrip::invalid_; } 
  if ( tmp.fedFe_ == SiStripFedKey::fedFeMask_ ) { tmp.fedFe_ = sistrip::invalid_; } 
  if ( tmp.fedCh_ == SiStripFedKey::fedChMask_ ) { tmp.fedCh_ = sistrip::invalid_; } 
  return tmp;  
}

// -----------------------------------------------------------------------------
//
std::ostream& operator<< ( std::ostream& os, const SiStripFedKey::Path& path ) {
  return os << "[SiStripFedKey::Path]"
	    << std::hex
	    << " key=0x" << std::setfill('0') << std::setw(8) << SiStripFedKey::key(path)
	    << std::dec
	    << " fedId=" << path.fedId_
	    << " fedFe=" << path.fedFe_
	    << " fedCh=" << path.fedCh_
	    << " channel=" << (12*path.fedFe_+path.fedCh_);
}
