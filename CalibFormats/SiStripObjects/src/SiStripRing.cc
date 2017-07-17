
#include "CalibFormats/SiStripObjects/interface/SiStripRing.h"
#include <iostream>

// -----------------------------------------------------------------------------
//
SiStripRing::SiStripRing( const FedChannelConnection& conn )
  : fecRing_( conn.fecRing() ), 
    ccus_()
{ 
  ccus_.reserve(256);
  addDevices( conn ); 
}

// -----------------------------------------------------------------------------
//
void SiStripRing::addDevices( const FedChannelConnection& conn ) {
  auto iccu = ccus_.begin();
  while ( iccu != ccus_.end() && (*iccu).ccuAddr() != conn.ccuAddr() ) { iccu++; }
  if ( iccu == ccus().end() ) { 
    ccus_.push_back( SiStripCcu( conn ) ); 
  } else { 
    iccu->addDevices( conn ); 
  }
}

