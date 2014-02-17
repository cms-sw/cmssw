// Last commit: $Id: SiStripFec.cc,v 1.7 2008/01/22 18:44:27 muzaffar Exp $

#include "CalibFormats/SiStripObjects/interface/SiStripFec.h"
#include <iostream>

// -----------------------------------------------------------------------------
//
SiStripFec::SiStripFec( const FedChannelConnection& conn )
  : fecSlot_( conn.fecSlot() ), 
    rings_() 
{ 
  rings_.reserve(8);
  addDevices( conn ); 
}

// -----------------------------------------------------------------------------
//
void SiStripFec::addDevices( const FedChannelConnection& conn ) {
  std::vector<SiStripRing>::const_iterator iring = rings().begin();
  while ( iring != rings().end() && (*iring).fecRing() != conn.fecRing() ) { iring++; }
  if ( iring == rings().end() ) { 
    rings_.push_back( SiStripRing( conn ) ); 
  } else { 
    const_cast<SiStripRing&>(*iring).addDevices( conn ); 
  }
}
