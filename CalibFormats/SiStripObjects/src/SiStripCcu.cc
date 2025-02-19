// Last commit: $Id: SiStripCcu.cc,v 1.7 2008/01/22 18:44:27 muzaffar Exp $

#include "CalibFormats/SiStripObjects/interface/SiStripCcu.h"
#include <iostream>
  
// -----------------------------------------------------------------------------
//
SiStripCcu::SiStripCcu( const FedChannelConnection& conn ) 
  : ccuAddr_( conn.ccuAddr() ), 
    modules_() 
{ 
  modules_.reserve(32);
  addDevices( conn ); 
}

// -----------------------------------------------------------------------------
//
void SiStripCcu::addDevices( const FedChannelConnection& conn ) {
  std::vector<SiStripModule>::const_iterator imod = modules().begin();
  while ( imod != modules().end() && (*imod).ccuChan() != conn.ccuChan() ) { imod++; }
  if ( imod == modules().end() ) { 
    modules_.push_back( SiStripModule( conn ) ); 
  } else { 
    const_cast<SiStripModule&>(*imod).addDevices( conn ); 
  }
}
