// Last commit: $Id: SiStripCcu.cc,v 1.5 2007/03/21 09:54:21 bainbrid Exp $

#include "CalibFormats/SiStripObjects/interface/SiStripCcu.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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
