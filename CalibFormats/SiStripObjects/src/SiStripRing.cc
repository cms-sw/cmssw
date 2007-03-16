#include "CalibFormats/SiStripObjects/interface/SiStripRing.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

// -----------------------------------------------------------------------------
//
void SiStripRing::addDevices( const FedChannelConnection& conn ) {
  std::vector<SiStripCcu>::const_iterator iccu = ccus().begin();
  while ( iccu != ccus().end() && (*iccu).ccuAddr() != conn.ccuAddr() ) { iccu++; }
  if ( iccu == ccus().end() ) { 
    ccus_.push_back( SiStripCcu( conn ) ); 
  } else { 
    const_cast<SiStripCcu&>(*iccu).addDevices( conn ); 
  }
}

