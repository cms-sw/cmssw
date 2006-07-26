#include "CalibFormats/SiStripObjects/interface/SiStripFec.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace std;

// -----------------------------------------------------------------------------
//
void SiStripFec::addDevices( const FedChannelConnection& conn ) {
  vector<SiStripRing>::const_iterator iring = rings().begin();
  while ( iring != rings().end() && (*iring).fecRing() != conn.fecRing() ) { iring++; }
  if ( iring == rings().end() ) { 
    rings_.push_back( SiStripRing( conn ) ); 
  } else { 
    const_cast<SiStripRing&>(*iring).addDevices( conn ); 
  }
}
