#include "CalibFormats/SiStripObjects/interface/SiStripCcu.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace std;

// -----------------------------------------------------------------------------
//
void SiStripCcu::addDevices( const FedChannelConnection& conn ) {
  vector<SiStripModule>::const_iterator imod = modules().begin();
  while ( imod != modules().end() && (*imod).ccuChan() != conn.ccuChan() ) { imod++; }
  if ( imod == modules().end() ) { 
    //     LogDebug("FecCabling") << "[SiStripCcu::addDevices]" 
    // 			   << " Adding new module with address " 
    // 			   << conn.ccuChan();
    modules_.push_back( SiStripModule( conn ) ); 
  } else { 
    //     LogDebug("FecCabling") << "[SiStripRing::addDevices]" 
    // 			   << " Module already exists with address " 
    // 			   << imod->ccuChan();
    const_cast<SiStripModule&>(*imod).addDevices( conn ); 
  }
}
