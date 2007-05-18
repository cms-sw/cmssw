// Last commit: $Id: SiStripFecCrate.cc,v 1.5 2007/03/21 09:54:21 bainbrid Exp $

#include "CalibFormats/SiStripObjects/interface/SiStripFecCrate.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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


