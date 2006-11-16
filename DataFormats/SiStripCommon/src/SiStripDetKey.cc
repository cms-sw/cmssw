#include "DataFormats/SiStripCommon/interface/SiStripDetKey.h"
#include <iomanip>

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
std::ostream& operator<< ( std::ostream& os, const SiStripDetKey::Path& path ) {
  return os << "[SiStripDetKey::Path]"
	    << std::hex
	    << " key=0x" << std::setfill('0') << std::setw(8) << SiStripDetId(path.detId_,path.apvPair_).rawId()
	    << " detId=0x" << std::setfill('0') << std::setw(8) << path.detId_ 
	    << std::dec
	    << " apvPair=" << path.apvPair_;
}
