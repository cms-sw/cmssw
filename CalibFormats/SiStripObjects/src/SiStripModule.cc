#include "CalibFormats/SiStripObjects/interface/SiStripModule.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>
#include <sstream>

using namespace std;

// -----------------------------------------------------------------------------
//
void SiStripModule::addDevices( const FedChannelConnection& conn ) {
  
  // Consistency check with HW addresses
  if ( fecCrate_ && fecCrate_ != conn.fecCrate() ) {
    edm::LogError("FecCabling") << "[SiStripFecCabling::addDevices]" 
				<< " Unexpected FEC crate ("
				<< conn.fecCrate() << ") for this module ("
				<< fecCrate_ << ")!";
    return;
  }
  if ( fecSlot_ && fecSlot_ != conn.fecSlot() ) {
    edm::LogError("FecCabling") << "[SiStripFecCabling::addDevices]" 
				<< " Unexpected FEC slot ("
				<< conn.fecSlot() << ") for this module ("
				<< fecSlot_ << ")!";
    return;
  }
  if ( fecRing_ && fecRing_ != conn.fecRing() ) {
    edm::LogError("FecCabling") << "[SiStripFecCabling::addDevices]" 
				<< " Unexpected FEC ring ("
				<< conn.fecRing() << ") for this module ("
				<< fecRing_ << ")!";
    return;
  }
  if ( ccuAddr_ && ccuAddr_ != conn.ccuAddr() ) {
    edm::LogError("FecCabling") << "[SiStripFecCabling::addDevices]" 
				<< " Unexpected CCU addr ("
				<< conn.ccuAddr() << ") for this module ("
				<< ccuAddr_ << ")!";
    return;
  }
  if ( ccuChan_ && ccuChan_ != conn.ccuChan() ) {
    edm::LogError("FecCabling") << "[SiStripFecCabling::addDevices]" 
				<< " Unexpected CCU chan ("
				<< conn.ccuChan() << ") for this module ("
				<< ccuChan_ << ")!";
    return;
  }

  // APVs
  addApv( conn.i2cAddr(0) );
  addApv( conn.i2cAddr(1) );

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
    edm::LogError("FecCabling") << "[SiStripFecCabling::activeApv]" 
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
    edm::LogWarning("FecCabling") << "[SiStripFecCabling::addApv]" 
				  << " Null APV I2C address!"; 
    return;
  } else if ( apv_address < 32 && apv_address > 37 ) {
    edm::LogError("FecCabling") << "[SiStripFecCabling::addApv]" 
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
  else { 
    edm::LogError("FecCabling") << "[SiStripFecCabling::addApv]" 
				<< "APV with I2C address " 
				<< apv_address
				<< " already exists!"; 
  }
  
  if ( added_apv ) { 
    stringstream ss;
    ss << "[SiStripModule::addApv] Added new APV with HW addresses:"
       << " FecCrate/FecSlot/CcuAddr/CcuChan/I2cAddr: "
       << fecCrate_ << "/"
       << fecSlot_ << "/"
       << fecRing_ << "/"
       << ccuAddr_ << "/"
       << ccuChan_ << "/"
       << apv_address;
    LogDebug("FecCabling") << ss.str();
  } 
  
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
    edm::LogError("FecCabling") << "[SiStripModule::nApvPairs]"
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
    edm::LogError("FecCabling") << "[SiStripFecCabling::lldChannel]"
				<< " Unexpected nunber of APV pairs!";
    return 0;
  }
  if ( nApvPairs_ == 2 && apv_pair_num == 1 ) { return 2; }
  else if ( nApvPairs_ == 2 && apv_pair_num == 3 ) { 
    edm::LogError("FecCabling") << "[SiStripFecCabling::lldChannel]"
				<< " Unexpected APV pair number!";
    return 0;
  } else { return apv_pair_num; } // is identical in this case
}

// -----------------------------------------------------------------------------
//
uint16_t SiStripModule::apvPairNum( const uint16_t& lld_channel ) const {
  if ( nApvPairs_ != 2 && nApvPairs_ != 3 ) {
    edm::LogError("FecCabling") << "[SiStripFecCabling::apvPairNum]"
				<< " Unexpected nunber of APV pairs!";
    return 0;
  }
  if ( nApvPairs_ == 2 && lld_channel == 2 ) { return 1; }
  else if ( nApvPairs_ == 2 && lld_channel == 1 ) { 
    edm::LogError("FecCabling") << "[SiStripFecCabling::apvPairNum]"
				<< " Unexpected LLD channel!";
    return 0;
  } else { return lld_channel; } // is identical in this case
}

// -----------------------------------------------------------------------------
//
const SiStripModule::FedChannel& SiStripModule::fedCh( const uint16_t& apv_pair ) const {
  static const FedChannel fed_ch = FedChannel(0,0);
  if ( !nApvPairs() ) {
    edm::LogError("FecCabling") << "[SiStripModule::fedCh] No APV pairs exist!";
    return fed_ch; 
  } else {
    uint16_t lld_ch;
    if ( nApvPairs() == 2 ) {
      if      ( apv_pair == 0 ) { lld_ch = 0; }
      else if ( apv_pair == 1 ) { lld_ch = 2; }
      else { 
	edm::LogError("FecCabling") << "[SiStripModule::fedCh] Unexpected pair number! " << apv_pair;
      }
    } else if ( nApvPairs() == 3 ) {
      if      ( apv_pair == 0 ) { lld_ch = 0; }
      else if ( apv_pair == 1 ) { lld_ch = 1; }
      else if ( apv_pair == 2 ) { lld_ch = 2; }
      else { 
	edm::LogError("FecCabling") << "[SiStripModule::fedCh] Unexpected pair number! " << apv_pair;
      }
    } else {
      edm::LogError("FecCabling") << "[SiStripModule::fedCh] Unexpected number of APV pairs: " << nApvPairs();
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
    edm::LogError("FecCabling") << "[SiStripModule::fedCh]" 
				<< " Unexpected I2C address (" 
				<< apv_address << ") for APV!"; 
    return false;
  }
  // Search for entry in map
  FedCabling::iterator ipair = cabling_.find( lld_ch );
  if ( ipair == cabling_.end() ) { cabling_[lld_ch] = fed_ch; }
  else { ipair->second = fed_ch; }
  return true;
}

// -----------------------------------------------------------------------------
//
void SiStripModule::print() const {
  stringstream ss;
  ss << "[SiStripModule::print]"
     << "  FecCrate/FecSlot/CcuAddr/CcuChan: "
     << "?/" // << fecCrate() << "/"
     << "?/" // << fecSlot() << "/"
     << "?/" // << fecRing() << "/"
     << "?/" // << ccuAddr() << "/"
     << this->ccuChan();
  ss << "  nApvs/apvAddrs: "
     << activeApvs().size() << "/";
  for ( uint16_t iapv = 0; iapv < activeApvs().size(); iapv++ ) {
    ss << activeApvs()[iapv];
    if ( activeApvs().size()-iapv > 1 ) { ss << "/"; }
  }
  ss << "  DCU/MUX/PLL/LLD: "
     << dcu() << "/"
     << mux() << "/"
     << pll() << "/"
     << lld() 
     << "  DcuId/DetId/nPairs: "
     << hex
     << setfill('0') << setw(8) << dcuId() << "/"
     << setfill('0') << setw(8) << detId() << "/"
     << dec
     << nApvPairs();
  ss << "  nConnected/apvAddr-FedId-FedCh: " 
     << fedChannels().size() << "/";
  FedCabling::const_iterator iconn;
  for ( iconn = fedChannels().begin(); iconn != fedChannels().end(); iconn++ ) {
    ss << iconn->first << "-"
       << iconn->second.first << "-"
       << iconn->second.second << "/";
  }
  LogDebug("FedCabling") << ss.str();
}

