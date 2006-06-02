#include "CalibFormats/SiStripObjects/interface/SiStripFecCrate.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace std;

// -----------------------------------------------------------------------------
//
void SiStripFecCrate::addDevices( const FedChannelConnection& conn ) {
  vector<SiStripFec>::const_iterator ifec = fecs().begin();
  while ( ifec != fecs().end() && (*ifec).fecSlot() != conn.fecSlot() ) { ifec++; }
  if ( ifec == fecs().end() ) { 
    LogDebug("FecCabling") << "[SiStripFecCrate::addDevices]" 
			   << " Adding new FEC with slot address " 
			   << conn.fecSlot();
    fecs_.push_back( SiStripFec( conn ) ); 
  } else { 
    LogDebug("FecCabling") << "[SiStripFecCrate::addDevices]" 
			   << " FEC already exists with slot address " 
			   << ifec->fecSlot();
    const_cast<SiStripFec&>(*ifec).addDevices( conn ); 
  }
}


