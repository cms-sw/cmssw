// Last commit: $Id: SiStripRing.cc,v 1.7 2008/01/22 18:44:27 muzaffar Exp $

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
  std::vector<SiStripCcu>::const_iterator iccu = ccus().begin();
  while ( iccu != ccus().end() && (*iccu).ccuAddr() != conn.ccuAddr() ) { iccu++; }
  if ( iccu == ccus().end() ) { 
    ccus_.push_back( SiStripCcu( conn ) ); 
  } else { 
    const_cast<SiStripCcu&>(*iccu).addDevices( conn ); 
  }
}

