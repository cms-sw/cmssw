// Last commit: $Id: SiStripRing.cc,v 1.6 2007/03/28 09:13:33 bainbrid Exp $

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

