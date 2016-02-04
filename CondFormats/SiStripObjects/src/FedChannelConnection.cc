#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iomanip>
#include <string>

using namespace sistrip;

// -----------------------------------------------------------------------------
// 
FedChannelConnection::FedChannelConnection( const uint16_t& fec_crate, 
					    const uint16_t& fec_slot, 
					    const uint16_t& fec_ring, 
					    const uint16_t& ccu_addr, 
					    const uint16_t& ccu_chan, 
					    const uint16_t& apv0,
					    const uint16_t& apv1,
					    const uint32_t& dcu_id,
					    const uint32_t& det_id,
					    const uint16_t& pairs ,
					    const uint16_t& fed_id,
					    const uint16_t& fed_ch,
					    const uint16_t& length,
					    const bool& dcu,
					    const bool& pll,
					    const bool& mux,
					    const bool& lld ) :
  fecCrate_(fec_crate), 
  fecSlot_(fec_slot), 
  fecRing_(fec_ring), 
  ccuAddr_(ccu_addr), 
  ccuChan_(ccu_chan),
  apv0_(apv0), 
  apv1_(apv1),
  dcuId_(dcu_id), 
  detId_(det_id), 
  nApvPairs_(pairs), 
  fedCrate_(sistrip::invalid_),
  fedSlot_(sistrip::invalid_),
  fedId_(fed_id), 
  fedCh_(fed_ch), 
  length_(length),
  dcu0x00_(dcu), 
  mux0x43_(mux), 
  pll0x44_(pll), 
  lld0x60_(lld)
{;}

// -----------------------------------------------------------------------------
// 
FedChannelConnection::FedChannelConnection() :
  fecCrate_(sistrip::invalid_), 
  fecSlot_(sistrip::invalid_), 
  fecRing_(sistrip::invalid_), 
  ccuAddr_(sistrip::invalid_), 
  ccuChan_(sistrip::invalid_),
  apv0_(sistrip::invalid_), 
  apv1_(sistrip::invalid_),
  dcuId_(sistrip::invalid32_), 
  detId_(sistrip::invalid32_), 
  nApvPairs_(sistrip::invalid_), 
  fedCrate_(sistrip::invalid_),
  fedSlot_(sistrip::invalid_),
  fedId_(sistrip::invalid_), 
  fedCh_(sistrip::invalid_), 
  length_(sistrip::invalid_),
  dcu0x00_(false), 
  mux0x43_(false), 
  pll0x44_(false), 
  lld0x60_(false)
{;}

// -----------------------------------------------------------------------------
//
bool operator< ( const FedChannelConnection& conn1, const FedChannelConnection& conn2 ) { 
  if ( conn1.fedId() < conn2.fedId() ) { return true; }
  else if ( conn1.fedId() == conn2.fedId() ) { return ( conn1.fedCh() < conn2.fedCh() ? true : false ); }
  else { return false; }
}

// -----------------------------------------------------------------------------
//
const uint16_t& FedChannelConnection::i2cAddr( const uint16_t& apv ) const { 
  if      ( apv == 0 ) { return apv0_; }
  else if ( apv == 1 ) { return apv1_; }
  else {
    if ( edm::isDebugEnabled() ) {
      edm::LogWarning(mlCabling_)
	<< "[FedChannelConnection::" << __func__ << "]"
	<< " Unexpected APV I2C address!" << apv;
    }
    static const uint16_t i2c_addr = 0;
    return i2c_addr;
  }
}

// -----------------------------------------------------------------------------
//
uint16_t FedChannelConnection::lldChannel() const {
  if      ( apv0_ == 32 || apv1_ == 33 ) { return 1; }
  else if ( apv0_ == 34 || apv1_ == 35 ) { return 2; }
  else if ( apv0_ == 36 || apv1_ == 37 ) { return 3; }
  else if ( apv0_ != sistrip::invalid_ ||
	    apv1_ != sistrip::invalid_ ) {
    if ( edm::isDebugEnabled() ) {
      edm::LogWarning(mlCabling_)
	<< "[FedChannelConnection::" << __func__ << "]"
	<< " Unexpected APV I2C addresses!" 
	<< " Apv0: " << apv0_
	<< " Apv1: " << apv1_;
    }
  }
  return sistrip::invalid_;
}

// -----------------------------------------------------------------------------
/** */
uint16_t FedChannelConnection::apvPairNumber() const {
  if ( nApvPairs_ == 2 ) {
    if      ( apv0_ == 32 || apv1_ == 33 ) { return 0; }
    else if ( apv0_ == 36 || apv1_ == 37 ) { return 1; }
    else { 
      if ( edm::isDebugEnabled() ) {
	edm::LogWarning(mlCabling_)
	  << "[FedChannelConnection::" << __func__ << "]"
	  << " APV I2C addresses (" 
	  << apv0_ << "/" << apv1_
	  << ") are incompatible with"
	  << " number of APV pairs (" 
	  << nApvPairs_ << ") found for this module!";
      }
    }
  } else if ( nApvPairs_ == 3 ) {
    if      ( apv0_ == 32 || apv1_ == 33 ) { return 0; }
    else if ( apv0_ == 34 || apv1_ == 35 ) { return 1; }
    else if ( apv0_ == 36 || apv1_ == 37 ) { return 2; }
    else { 
      if ( edm::isDebugEnabled() ) {
	edm::LogWarning(mlCabling_)
	  << "[FedChannelConnection::" << __func__ << "]"
	  << " APV I2C addresses (" 
	  << apv0_ << "/" << apv1_
	  << ") are incompatible with"
	  << " number of APV pairs (" 
	  << nApvPairs_ << ") found for this module!";
      }
    }
  } else { 
    if ( edm::isDebugEnabled() ) {
      edm::LogWarning(mlCabling_) 
	<< "[FedChannelConnection::" << __func__ << "]"
	<< " Unexpected number of APV pairs: " << nApvPairs_;
    }
  }
  return sistrip::invalid_;
}

// -----------------------------------------------------------------------------
// 
void FedChannelConnection::print( std::stringstream& ss ) const {
  ss << " [FedChannelConnection::" << __func__ << "]" << std::endl
     << " FedCrate/FedSlot/FedId/FeUnit/FeChan/FedCh : "
     << fedCrate() << "/"
     << fedSlot() << "/"
     << fedId() << "/"
     << SiStripFedKey::feUnit( fedCh() ) << "/" 
     << SiStripFedKey::feChan( fedCh() ) << "/" 
     << fedCh() << std::endl
     << " FecCrate/FecSlot/FecRing/CcuAddr/CcuChan   : "
     << fecCrate() << "/"
     << fecSlot() << "/"
     << fecRing() << "/"
     << ccuAddr() << "/"
     << ccuChan() << std::endl
     << " DcuId/DetId                                : "
     << std::hex
     << "0x" << std::setfill('0') << std::setw(8) << dcuId() << "/"
     << "0x" << std::setfill('0') << std::setw(8) << detId() << std::endl
     << std::dec
     << " LldChan/APV0/APV1                          : "
     << lldChannel() << "/" 
     << i2cAddr(0) << "/"
     << i2cAddr(1) << std::endl
     << " pairNumber/nPairs/nStrips                  : "
     << apvPairNumber() << "/"
     << nApvPairs() << "/"
     << 256*nApvPairs() << std::endl
     << " DCU/MUX/PLL/LLD found                      : "
     << std::boolalpha
     << dcu() << "/"
     << mux() << "/"
     << pll() << "/"
     << lld()
     << std::noboolalpha;
}

// -----------------------------------------------------------------------------
// 
void FedChannelConnection::terse( std::stringstream& ss ) const {
  ss << " FED:cr/sl/id/fe/ch/chan=" 
     << fedCrate() << "/" 
     << fedSlot() << "/" 
     << fedId() << "/" 
     << SiStripFedKey::feUnit( fedCh() ) << "/" 
     << SiStripFedKey::feChan( fedCh() ) << "/" 
     << fedCh() << "," 
     << " FEC:cr/sl/ring/ccu/mod="
     << fecCrate() << "/"
     << fecSlot() << "/" 
     << fecRing() << "/"
     << ccuAddr() << "/" 
     << ccuChan() << ","
     << " apvs=" 
     << i2cAddr(0) << "/" 
     << i2cAddr(1) << "," 
     << " pair=" << apvPairNumber()+1
     << " (from " << nApvPairs() << "),"
     << " dcu/detid=" 
     << std::hex
     << "0x" << std::setfill('0') << std::setw(8) << dcuId() << "/"
     << "0x" << std::setfill('0') << std::setw(8) << detId() 
     << std::dec;
}

// -----------------------------------------------------------------------------
//
std::ostream& operator<< ( std::ostream& os, const FedChannelConnection& conn ) {
  std::stringstream ss;
  conn.print(ss);
  os << ss.str();
  return os;
}

