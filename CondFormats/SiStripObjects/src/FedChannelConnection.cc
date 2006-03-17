#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include <iostream>
#include <string>

// -----------------------------------------------------------------------------
//
uint16_t FedChannelConnection::lldChannel() const {
  if      ( apv0_ == 32 && apv1_ == 33 ) { return 0; }
  else if ( apv0_ == 34 && apv1_ == 35 ) { return 1; }
  else if ( apv0_ == 36 && apv1_ == 37 ) { return 2; }
  else {
    std::cerr << "[FedChannelConnection::lldChannel]"
	      << " Unexpected APV I2C addresses!" 
	      << " Apv0: " << apv0_
	      << " Apv1: " << apv1_
	      << std::endl;
  }
  return 0;
}

// -----------------------------------------------------------------------------
/** */
uint16_t FedChannelConnection::apvPairNumber() const {
  if ( nApvPairs_ == 2 ) {
    if ( apv0_ == 32 ) { return 0; }
    else if ( apv0_ == 36 ) { return 1; }
    else { 
      std::cerr << "[FedChannelConnection::apvPairNumber]"
		<< " Incompatible data!" 
		<< " Number of APV pairs: " << nApvPairs_
		<< " I2C address of APV0: " << apv0_
		<< std::endl;
    }
  } else if ( nApvPairs_ == 3 ) {
    if ( apv0_ == 32 ) { return 0; }
    else if ( apv0_ == 34 ) { return 1; }
    else if ( apv0_ == 36 ) { return 2; }
    else { 
      std::cerr << "[FedChannelConnection::apvPairNumber]"
		<< " Incompatible data!" 
		<< " Number of APV pairs: " << nApvPairs_
		<< " I2C address of APV0: " << apv0_
		<< std::endl;
    }
  } else {
    std::cerr << "[FedChannelConnection::apvPairNumber]"
	      << " Unexpected number of APV pairs: " << nApvPairs_
	      << std::endl;
  }
  return 0;
}

// -----------------------------------------------------------------------------
/** */
void FedChannelConnection::print() const {
  std::cout << "[FedChannelConnection::print]"
	    << "  FecCrate/FecSlot/CcuAddr/CcuChan/APV0/APV1: "
	    << fecCrate() << "/"
	    << fecSlot() << "/"
	    << fecRing() << "/"
	    << ccuAddr() << "/"
	    << ccuChan() << "/"
	    << i2cAddrApv0() << "/"
	    << i2cAddrApv1() 
	    << "  DCU/MUX/PLL/LLD: "
	    << dcu() << "/"
	    << mux() << "/"
	    << pll() << "/"
	    << lld() 
	    << "  DcuId/DetId/nPairs: "
	    << dcuId() << "/"
	    << detId() << "/"
	    << nApvPairs() << "/"
	    << "  LldChan/PairNumber: "
	    << lldChannel() << "/"
	    << apvPairNumber() 
	    << "  FedId/FedCh: "
	    << fedId() << "/"
	    << fedCh() << std::endl;
}
