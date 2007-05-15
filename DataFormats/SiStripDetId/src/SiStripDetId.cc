#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include <iomanip>

// -----------------------------------------------------------------------------
//
SiStripDetId::SiStripDetId() : DetId() {;}

// -----------------------------------------------------------------------------
//
SiStripDetId::SiStripDetId( const uint32_t& raw_id ) : DetId( raw_id ) {;}

// -----------------------------------------------------------------------------
//
SiStripDetId::SiStripDetId( const DetId& det_id ) : DetId( det_id.rawId() ) {;}

// -----------------------------------------------------------------------------
//
SiStripDetId::SiStripDetId( Detector det, int subdet ) : DetId( det, subdet ) {;}

// -----------------------------------------------------------------------------
//
SiStripDetId::SiStripDetId( const uint32_t& raw_id, 
			    const uint16_t& reserved )
  : DetId( raw_id ) 
{
  id_ &= ( ~static_cast<uint32_t>(reservedMask_<<reservedStartBit_) );
  id_ |= ( ( reserved & reservedMask_ ) << reservedStartBit_ );
}

// -----------------------------------------------------------------------------
//
SiStripDetId::SiStripDetId( const DetId& det_id, 
			    const uint16_t& reserved )
  : DetId( det_id.rawId() ) 
{
  id_ &= ( ~static_cast<uint32_t>(reservedMask_<<reservedStartBit_) );
  id_ |= ( ( reserved & reservedMask_ ) << reservedStartBit_ );
}

// -----------------------------------------------------------------------------
//
std::ostream& operator<< ( std::ostream& os, const SiStripDetId& id ) {
  return os << "[SiStripDetId]"
	    << std::hex
	    << " rawId=0x" << std::setfill('0') << std::setw(8) << id.rawId()
	    << std::dec
	    << " detId=" << id.det() 
	    << " subDetId=" << id.subdetId()
	    << " reserved=" << id.reserved()
	    << std::hex
	    << " bits[0:24]=" << std::setfill('0') << std::setw(8) << (0x01FFFFFF & id.rawId())
	    << std::dec;
}

