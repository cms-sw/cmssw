#include "CalibFormats/SiStripObjects/interface/SiStripModule.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>
#include <sstream>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
//
void SiStripModule::addDevices( const FedChannelConnection& conn ) {
  
  // Consistency check with HW addresses
  if ( path_.fecCrate_ && path_.fecCrate_ != conn.fecCrate() ) {
    edm::LogWarning(mlCabling_)
      << "SiStripModule::" << __func__ << "]"
      << " Unexpected FEC crate ("
      << conn.fecCrate() << ") for this module ("
      << path_.fecCrate_ << ")!";
    return;
  }
  if ( path_.fecSlot_ && path_.fecSlot_ != conn.fecSlot() ) {
    edm::LogWarning(mlCabling_)
            << "SiStripModule::" << __func__ << "]"
	    << " Unexpected FEC slot ("
	    << conn.fecSlot() << ") for this module ("
	    << path_.fecSlot_ << ")!";
    return;
  }
  if ( path_.fecRing_ && path_.fecRing_ != conn.fecRing() ) {
    edm::LogWarning(mlCabling_)
      << "SiStripModule::" << __func__ << "]"
      << " Unexpected FEC ring ("
      << conn.fecRing() << ") for this module ("
      << path_.fecRing_ << ")!";
    return;
  }
  if ( path_.ccuAddr_ && path_.ccuAddr_ != conn.ccuAddr() ) {
    edm::LogWarning(mlCabling_)
      << "SiStripModule::" << __func__ << "]"
      << " Unexpected CCU addr ("
      << conn.ccuAddr() << ") for this module ("
      << path_.ccuAddr_ << ")!";
    return;
  }
  if ( path_.ccuChan_ && path_.ccuChan_ != conn.ccuChan() ) {
    edm::LogWarning(mlCabling_)
      << "SiStripModule::" << __func__ << "]"
      << " Unexpected CCU chan ("
      << conn.ccuChan() << ") for this module ("
      << path_.ccuChan_ << ")!";
    return;
  }

  // APVs
  if ( conn.i2cAddr(0) ) { addApv( conn.i2cAddr(0) ); }
  if ( conn.i2cAddr(1) ) { addApv( conn.i2cAddr(1) ); }
  
  // Detector
  dcuId( conn.dcuId() ); 
  detId( conn.detId() ); 
  nApvPairs( conn.nApvPairs() ); 
  
  // FED cabling
  FedChannel fed_ch = FedChannel( conn.fedId(), conn.fedCh() ); 
  fedCh( conn.i2cAddr(0), fed_ch );
  
  // DCU, MUX, PLL, LLD
  if ( conn.dcu() ) { dcu0x00_ = true; }
  if ( conn.mux() ) { mux0x43_ = true; }
  if ( conn.pll() ) { pll0x44_ = true; }
  if ( conn.lld() ) { lld0x60_ = true; }
  
}

// -----------------------------------------------------------------------------
//
vector<uint16_t> SiStripModule::activeApvs() const {
  vector<uint16_t> apvs;
  if ( apv0x32_ ) { apvs.push_back( apv0x32_ ); }
  if ( apv0x33_ ) { apvs.push_back( apv0x33_ ); }
  if ( apv0x34_ ) { apvs.push_back( apv0x34_ ); }
  if ( apv0x35_ ) { apvs.push_back( apv0x35_ ); }
  if ( apv0x36_ ) { apvs.push_back( apv0x36_ ); }
  if ( apv0x37_ ) { apvs.push_back( apv0x37_ ); }
  return apvs;
}

// -----------------------------------------------------------------------------
//
 const uint16_t& SiStripModule::activeApv( const uint16_t& apv_address ) const {
  if      ( apv_address == 0 || apv_address == 32 ) { return apv0x32_; }
  else if ( apv_address == 1 || apv_address == 33 ) { return apv0x33_; }
  else if ( apv_address == 2 || apv_address == 34 ) { return apv0x34_; }
  else if ( apv_address == 3 || apv_address == 35 ) { return apv0x35_; }
  else if ( apv_address == 4 || apv_address == 36 ) { return apv0x36_; }
  else if ( apv_address == 5 || apv_address == 37 ) { return apv0x37_; }
  else {
    edm::LogWarning(mlCabling_)
      << "SiStripModule::" << __func__ << "]"
      << " Unexpected I2C address or number (" 
      << apv_address << ") for this module!";
  }
  static const uint16_t address = 0;
  return address;
}

// -----------------------------------------------------------------------------
//
void SiStripModule::addApv( const uint16_t& apv_address ) {

  // Some checks on value of APV I2C address
  if ( apv_address == 0 ) {
    edm::LogWarning(mlCabling_)
      << "SiStripModule::" << __func__ << "]"
      << " Null APV I2C address!"; 
    return;
  } else if ( apv_address < 32 && apv_address > 37 ) {
    edm::LogWarning(mlCabling_)
            << "SiStripModule::" << __func__ << "]"
	    << " Unexpected I2C address (" 
	    << apv_address << ") for APV!"; 
    return;
  }

  bool added_apv = false; 
  if      ( !apv0x32_ && apv_address == 32 ) { apv0x32_ = 32; added_apv = true; }
  else if ( !apv0x33_ && apv_address == 33 ) { apv0x33_ = 33; added_apv = true; }
  else if ( !apv0x34_ && apv_address == 34 ) { apv0x34_ = 34; added_apv = true; }
  else if ( !apv0x35_ && apv_address == 35 ) { apv0x35_ = 35; added_apv = true; }
  else if ( !apv0x36_ && apv_address == 36 ) { apv0x36_ = 36; added_apv = true; }
  else if ( !apv0x37_ && apv_address == 37 ) { apv0x37_ = 37; added_apv = true; }
  
  stringstream ss;
  ss << "SiStripModule::" << __func__ << "]";
  if ( added_apv ) { ss << " Added new APV with"; }
  else { ss << " APV already exists with"; }
  ss << " FecCrate/FecSlot/CcuAddr/CcuChan/I2cAddr: "
     << path_.fecCrate_ << "/"
     << path_.fecSlot_ << "/"
     << path_.fecRing_ << "/"
     << path_.ccuAddr_ << "/"
     << path_.ccuChan_ << "/"
     << apv_address;
  //if ( added_apv ) { LogTrace(mlCabling_) << ss.str(); }
  /* else */ if ( !added_apv ) { edm::LogWarning(mlCabling_) << ss.str(); }
  
}

// -----------------------------------------------------------------------------
//
void SiStripModule::nApvPairs( const uint16_t& npairs ) { 
  if ( npairs == 2 || npairs == 3 ) { nApvPairs_ = npairs; } 
  else if ( npairs == 0 ) {
    nApvPairs_ = 0;
    if ( apv0x32_ || apv0x33_ ) { nApvPairs_++; }
    if ( apv0x34_ || apv0x35_ ) { nApvPairs_++; }
    if ( apv0x36_ || apv0x37_ ) { nApvPairs_++; }
  } else { 
    edm::LogWarning(mlCabling_)
      << "SiStripModule::" << __func__ << "]"
      << " Unexpected number of APV pairs: " 
      << npairs;
  }
} 

// -----------------------------------------------------------------------------
//
SiStripModule::FedChannel SiStripModule::activeApvPair( const uint16_t& lld_channel ) const {
  if      ( lld_channel == 0 ) { return FedChannel(apv0x32_,apv0x33_); }
  else if ( lld_channel == 1 ) { return FedChannel(apv0x34_,apv0x35_); }
  else if ( lld_channel == 2 ) { return FedChannel(apv0x36_,apv0x37_); }
  else                         { return FedChannel(0,0); }
}

// -----------------------------------------------------------------------------
//
uint16_t SiStripModule::lldChannel( const uint16_t& apv_pair_num ) const {
  if ( nApvPairs_ != 2 && nApvPairs_ != 3 ) {
    edm::LogWarning(mlCabling_)
      << "SiStripModule::" << __func__ << "]"
      << " Unexpected nunber of APV pairs!";
    return 0;
  }
  if ( nApvPairs_ == 2 && apv_pair_num == 1 ) { return 2; }
  else if ( nApvPairs_ == 2 && apv_pair_num == 3 ) { 
    edm::LogWarning(mlCabling_) << "[SiStripFecCabling::lldChannel]"
				<< " Unexpected APV pair number!";
    return 0;
  } else { return apv_pair_num; } // is identical in this case
}

// -----------------------------------------------------------------------------
//
uint16_t SiStripModule::apvPairNumber( const uint16_t& lld_channel ) const {
  if ( nApvPairs_ != 2 && nApvPairs_ != 3 ) {
    edm::LogWarning(mlCabling_)
      << "SiStripModule::" << __func__ << "]"
      << " Unexpected nunber of APV pairs!";
    return 0;
  }
  if ( nApvPairs_ == 2 && lld_channel == 2 ) { return 1; }
  else if ( nApvPairs_ == 2 && lld_channel == 1 ) { 
    edm::LogWarning(mlCabling_)
      << "SiStripModule::" << __func__ << "]"
      << " Unexpected LLD channel!";
    return 0;
  } else { return lld_channel; } // is identical in this case
}

// -----------------------------------------------------------------------------
//
const SiStripModule::FedChannel& SiStripModule::fedCh( const uint16_t& apv_pair ) const {
  static const FedChannel fed_ch = FedChannel(0,0);
  if ( !nApvPairs() ) {
    edm::LogWarning(mlCabling_)
      << "SiStripModule::" << __func__ << "]"
      << " No APV pairs exist!";
    return fed_ch; 
  } else {
    uint16_t lld_ch;
    if ( nApvPairs() == 2 ) {
      if      ( apv_pair == 0 ) { lld_ch = 0; }
      else if ( apv_pair == 1 ) { lld_ch = 2; }
      else { 
	edm::LogWarning(mlCabling_)
	  << "SiStripModule::" << __func__ << "]"
	  << " Unexpected pair number! " << apv_pair;
      }
    } else if ( nApvPairs() == 3 ) {
      if      ( apv_pair == 0 ) { lld_ch = 0; }
      else if ( apv_pair == 1 ) { lld_ch = 1; }
      else if ( apv_pair == 2 ) { lld_ch = 2; }
      else { 
	edm::LogWarning(mlCabling_)
	  << "SiStripModule::" << __func__ << "]"
	  << " Unexpected pair number! " << apv_pair;
      }
    } else {
      edm::LogWarning(mlCabling_) 
	<< "SiStripModule::" << __func__ << "]"
	<< " Unexpected number of APV pairs: " << nApvPairs();
    }
    FedCabling::const_iterator ipair = cabling_.find( lld_ch );
    if ( ipair != cabling_.end() ) { return (*ipair).second; }
    else { return fed_ch; }
  }
}

// -----------------------------------------------------------------------------
//
bool SiStripModule::fedCh( const uint16_t& apv_address, 
			   const FedChannel& fed_ch ) {
  // Determine LLD channel
  int16_t lld_ch = 0;
  if      ( apv_address == 32 || apv_address == 33 ) { lld_ch = 0; }
  else if ( apv_address == 34 || apv_address == 35 ) { lld_ch = 1; }
  else if ( apv_address == 36 || apv_address == 37 ) { lld_ch = 2; }
  else if ( apv_address == 0 ) { ; } //@@ do nothing?
  else { 
    edm::LogWarning(mlCabling_) << "[SiStripModule::fedCh]" 
				<< " Unexpected I2C address (" 
				<< apv_address << ") for APV!"; 
    return false;
  }
  // Search for entry in map
  //@@ use FedKey as key instead of lld chan? what about "duplicates"? always append to map? then can have >3 entries. useful for debug?
  FedCabling::iterator ipair = cabling_.find( lld_ch );
  if ( ipair == cabling_.end() ) { cabling_[lld_ch] = fed_ch; }
  else { ipair->second = fed_ch; }
  return true;
}

// -----------------------------------------------------------------------------
//
void SiStripModule::print( stringstream& ss ) const {

  ss << "[SiStripModule]" << endl
     << "  crate/FEC/CCU/Module: "
     << path().fecCrate_ << "/"
     << path().fecSlot_ << "/"
     << path().fecRing_ << "/"
     << path().ccuAddr_ << "/"
     << path().ccuChan_ << endl;

  ss << "  ActiveApvs: ";
  if ( activeApvs().empty() ) { ss << "NONE!"; }
  vector<uint16_t>::const_iterator iapv = activeApvs().begin();
  for ( ; iapv != activeApvs().end(); iapv++ ) {
    if ( *iapv ) { ss << *iapv << " "; }
  }
  ss << endl;

  ss << "  DcuId/DetId/nPairs: "
     << hex
     << "0x" << setfill('0') << setw(8) << dcuId() << "/"
     << "0x" << setfill('0') << setw(8) << detId() << "/"
     << dec
     << nApvPairs() << endl;

  ss << "  ApvPairNum/FedId/Ch: ";
  SiStripModule::FedCabling::const_iterator ichan = fedChannels().begin();
  for ( ; ichan != fedChannels().end(); ichan++ ) {
    if ( ichan->first ) {
      ss << ichan->first << "/"
	 << ichan->second.first << "/"
	 << ichan->second.second << " ";
    }
  }
  ss << endl;
  
  ss << "  DCU/MUX/PLL/LLD found: "
     << bool(dcu0x00_) << "/"
     << bool(mux0x43_) << "/"
     << bool(pll0x44_) << "/"
     << bool(lld0x60_);
  
}

// -----------------------------------------------------------------------------
//
ostream& operator<< ( ostream& os, const SiStripModule& device ) {
  stringstream ss;
  device.print(ss);
  os << ss.str();
  return os;
}
