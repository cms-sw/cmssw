#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include <iostream>
#include <string>

// -----------------------------------------------------------------------------
//
unsigned short FedChannelConnection::pairPos() const {
  if      ( apv0_ == 32 && apv1_ == 33 ) { return 0; }
  else if ( apv0_ == 34 && apv1_ == 35 ) { return 1; }
  else if ( apv0_ == 36 && apv1_ == 37 ) { return 2; }
  else {
    std::cerr << "[FedChannelConnection::pairPos]"
	      << " Unexpected APV I2C addresses!" 
	      << " Apv0: " << apv0_
	      << " Apv1: " << apv1_
	      << std::endl;
  }
  return 0;
}

// -----------------------------------------------------------------------------
//
unsigned short FedChannelConnection::pairId() const {
  if ( nPairs_ == 2 ) {
    if ( apv0_ == 32 ) { return 0; }
    else if ( apv0_ == 36 ) { return 1; }
    else { 
      std::cerr << "[FedChannelConnection::pairPos]"
		<< " Incompatible data!" 
		<< " Number of APV pairs: " << nPairs_
		<< " I2C address of APV0: " << apv0_
		<< std::endl;
    }
  } else if ( nPairs_ == 3 ) {
    if ( apv0_ == 32 ) { return 0; }
    else if ( apv0_ == 34 ) { return 1; }
    else if ( apv0_ == 36 ) { return 2; }
    else { 
      std::cerr << "[FedChannelConnection::pairPos]"
		<< " Incompatible data!" 
		<< " Number of APV pairs: " << nPairs_
		<< " I2C address of APV0: " << apv0_
		<< std::endl;
    }
  } else {
    std::cerr << "[FedChannelConnection::pairPos]"
	      << " Unexpected number of APV pairs: " << nPairs_
	      << std::endl;
  }
  return 0;
}
