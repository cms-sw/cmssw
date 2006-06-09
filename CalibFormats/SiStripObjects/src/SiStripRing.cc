#include "CalibFormats/SiStripObjects/interface/SiStripRing.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace std;

// -----------------------------------------------------------------------------
//
void SiStripRing::addDevices( const FedChannelConnection& conn ) {
  vector<SiStripCcu>::const_iterator iccu = ccus().begin();
  while ( iccu != ccus().end() && (*iccu).ccuAddr() != conn.ccuAddr() ) { iccu++; }
  if ( iccu == ccus().end() ) { 
    //     LogDebug("FecCabling") << "[SiStripRing::addDevices]" 
    // 			   << " Adding new CCU with address " 
    // 			   << conn.ccuAddr();
    ccus_.push_back( SiStripCcu( conn ) ); 
  } else { 
    //     LogDebug("FecCabling") << "[SiStripRing::addDevices]" 
    // 			   << " CCU already exists with address " 
    // 			   << iccu->ccuAddr();
    const_cast<SiStripCcu&>(*iccu).addDevices( conn ); 
  }
}

