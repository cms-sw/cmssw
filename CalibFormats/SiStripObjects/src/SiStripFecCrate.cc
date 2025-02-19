// Last commit: $Id: SiStripFecCrate.cc,v 1.7 2008/01/22 18:44:27 muzaffar Exp $

#include "CalibFormats/SiStripObjects/interface/SiStripFecCrate.h"
#include <iostream>

// -----------------------------------------------------------------------------
//
SiStripFecCrate::SiStripFecCrate( const FedChannelConnection& conn )
  : fecCrate_( conn.fecCrate() ), 
    fecs_() 
{ 
  fecs_.reserve(20);
  addDevices( conn ); 
}

// -----------------------------------------------------------------------------
//
void SiStripFecCrate::addDevices( const FedChannelConnection& conn ) {
  std::vector<SiStripFec>::const_iterator ifec = fecs().begin();
  while ( ifec != fecs().end() && (*ifec).fecSlot() != conn.fecSlot() ) { ifec++; }
  if ( ifec == fecs().end() ) { 
    fecs_.push_back( SiStripFec( conn ) ); 
  } else { 
    const_cast<SiStripFec&>(*ifec).addDevices( conn ); 
  }
}


