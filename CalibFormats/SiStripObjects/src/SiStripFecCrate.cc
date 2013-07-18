// Last commit: $Id: SiStripFecCrate.cc,v 1.6 2007/03/28 09:13:33 bainbrid Exp $

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


