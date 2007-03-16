#include "CalibFormats/SiStripObjects/interface/SiStripFecCrate.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

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


