#include "DataFormats/SiStripDetId/interface/SiStripReadoutKey.h"
#include <iostream>
#include <sstream>

// -----------------------------------------------------------------------------
// definition of static (public) data
const uint16_t SiStripReadoutKey::all_ = 0xFFFF;

// -----------------------------------------------------------------------------
//
uint32_t SiStripReadoutKey::key( uint16_t fed_id, 
				 uint16_t fed_ch ) {
  return ( static_cast<uint32_t>( ( fed_id & 0xFFFF ) << 16 ) | 
	   static_cast<uint32_t>( ( fed_ch & 0xFFFF ) <<  0 ) );
}

// -----------------------------------------------------------------------------
//
SiStripReadoutKey::ReadoutPath SiStripReadoutKey::path( uint32_t key ) {
  ReadoutPath path;
  path.fedId_ = (key>>16) & 0xFFFF;
  path.fedCh_ = (key>> 0) & 0xFFFF;
  if ( path.fedId_ == 0xFFFF ) { path.fedId_ = all_; } 
  if ( path.fedCh_ == 0xFFFF ) { path.fedCh_ = all_; } 
  return path;
}

