#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

// -----------------------------------------------------------------------------
//
uint16_t FedChannelConnection::lldChannel() const {
  if      ( apv0_ == 32 || apv1_ == 33 ) { return 0; }
  else if ( apv0_ == 34 || apv1_ == 35 ) { return 1; }
  else if ( apv0_ == 36 || apv1_ == 37 ) { return 2; }
  else {
    edm::LogWarning("Cabling") << "[FedChannelConnection::lldChannel]"
			       << " Unexpected APV I2C addresses!" 
			       << " Apv0: " << apv0_
			       << " Apv1: " << apv1_;
  }
  return 0;
}

// -----------------------------------------------------------------------------
/** */
uint16_t FedChannelConnection::apvPairNumber() const {
  if ( nApvPairs_ == 2 ) {
    if      ( apv0_ == 32 || apv1_ == 33 ) { return 0; }
    else if ( apv0_ == 36 || apv1_ == 37 ) { return 1; }
    else { 
      edm::LogWarning("Cabling") << "[FedChannelConnection::apvPairNumber]"
				 << " Incompatible data!" 
				 << " APV pairs: " << nApvPairs_
				 << " APV0: " << apv0_
				 << " APV1: " << apv1_;
    }
  } else if ( nApvPairs_ == 3 ) {
    if      ( apv0_ == 32 || apv1_ == 33 ) { return 0; }
    else if ( apv0_ == 34 || apv1_ == 35 ) { return 1; }
    else if ( apv0_ == 36 || apv1_ == 37 ) { return 2; }
    else { 
      edm::LogWarning("Cabling") << "[FedChannelConnection::apvPairNumber]"
				 << " Incompatible data!" 
				 << " APV pairs: " << nApvPairs_
				 << " APV0: " << apv0_
				 << " APV1: " << apv1_;
    }
  } else if ( nApvPairs_ == 0 ) {
    LogDebug("Cabling") << "[FedChannelConnection::apvPairNumber] Zero APV pairs";
  } else {
    edm::LogWarning("Cabling") << "[FedChannelConnection::apvPairNumber]"
			       << " Unexpected number of APV pairs: " << nApvPairs_;
  }
  return 0;
}

// -----------------------------------------------------------------------------
/** */
void FedChannelConnection::print() const {
  std::stringstream ss;
  ss << "[FedChannelConnection::print]"
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
     << std::hex
     << std::setfill('0') << std::setw(8) << dcuId() << "/"
     << std::setfill('0') << std::setw(8) << detId() << "/"
     << std::dec
     << nApvPairs() 
     << "  LldChan/PairNumber: "
     << lldChannel() << "/"
     << apvPairNumber() 
     << "  FedId/FedCh: "
     << fedId() << "/"
     << fedCh();
  LogDebug("Cabling") << ss.str();
}
