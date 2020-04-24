
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
  auto ifec = fecs_.begin();
  while ( ifec != fecs_.end() && (*ifec).fecSlot() != conn.fecSlot() ) { ifec++; }
  if ( ifec == fecs_.end() ) { 
    fecs_.push_back( SiStripFec( conn ) ); 
  } else { 
    ifec->addDevices( conn ); 
  }
}


