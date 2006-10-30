#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iomanip>
#include <sstream>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
//
const uint16_t& FedChannelConnection::i2cAddr( const uint16_t& apv ) const { 
  if      ( apv == 0 ) { return apv0_; }
  else if ( apv == 1 ) { return apv1_; }
  else {
    edm::LogWarning(mlCabling_)
      << "[FedChannelConnection::" << __func__ << "]"
      << " Unexpected APV I2C address!" << apv;
    static const uint16_t i2c_addr = 0;
    return i2c_addr;
  }
}

// -----------------------------------------------------------------------------
//
uint16_t FedChannelConnection::lldChannel() const {
  if      ( apv0_ == 32 || apv1_ == 33 ) { return 0; }
  else if ( apv0_ == 34 || apv1_ == 35 ) { return 1; }
  else if ( apv0_ == 36 || apv1_ == 37 ) { return 2; }
  else {
    edm::LogWarning(mlCabling_)
      << "[FedChannelConnection::" << __func__ << "]"
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
      edm::LogWarning(mlCabling_)
	<< "[FedChannelConnection::" << __func__ << "]"
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
      edm::LogWarning(mlCabling_)
	<< "[FedChannelConnection::" << __func__ << "]"
	<< " Incompatible data!" 
	<< " APV pairs: " << nApvPairs_
	<< " APV0: " << apv0_
	<< " APV1: " << apv1_;
    }
  } else {
    edm::LogWarning(mlCabling_) 
      << "[FedChannelConnection::" << __func__ << "]"
      << " Unexpected number of APV pairs: " << nApvPairs_;
  }
  return 0;
}

// -----------------------------------------------------------------------------
/** */
void FedChannelConnection::print( stringstream& ss ) const {
  ss << "  FedId/Ch: "
     << fedId() << "/"
     << fedCh() << endl
     << "  Crate/FEC/CCU/Module/LLDchan/APV0/1: "
     << fecCrate() << "/"
     << fecSlot() << "/"
     << fecRing() << "/"
     << ccuAddr() << "/"
     << ccuChan() << "/"
     << lldChannel() << "/"
     << i2cAddr(0) << "/"
     << i2cAddr(1) << endl
     << "  DcuId/DetId/nPairs/pairNum: "
     << hex
     << "0x" << setfill('0') << setw(8) << dcuId() << "/"
     << "0x" << setfill('0') << setw(8) << detId() << "/"
     << dec
     << nApvPairs() << "/"
     << apvPairNumber() << endl
     << "  DCU/MUX/PLL/LLD found: "
     << dcu() << "/"
     << mux() << "/"
     << pll() << "/"
     << lld();
}

// -----------------------------------------------------------------------------
//
ostream& operator<< ( ostream& os, const FedChannelConnection& conn ) {
  stringstream ss;
  conn.print(ss);
  os << ss.str();
  return os;
}

