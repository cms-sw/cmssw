// Last commit: $Id: SiStripModule.cc,v 1.19 2009/10/27 09:50:29 lowette Exp $

#include "CalibFormats/SiStripObjects/interface/SiStripModule.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripModule::SiStripModule( const FedChannelConnection& conn ) 
  : key_( conn.fecCrate(), 
	  conn.fecSlot(), 
	  conn.fecRing(), 
	  conn.ccuAddr(), 
	  conn.ccuChan() ),
    apv32_(0), 
    apv33_(0), 
    apv34_(0), 
    apv35_(0), 
    apv36_(0), 
    apv37_(0), 
    dcu0x00_(0), 
    mux0x43_(0), 
    pll0x44_(0), 
    lld0x60_(0), 
    dcuId_(0), 
    detId_(0), 
    nApvPairs_(0),
    cabling_(), 
    length_(0) 
{ 
  addDevices( conn ); 
}

// -----------------------------------------------------------------------------
//
void SiStripModule::addDevices( const FedChannelConnection& conn ) {
  
  if ( key_.fecCrate() && key_.fecCrate() != conn.fecCrate() ) {
    edm::LogWarning(mlCabling_)
      << "SiStripModule::" << __func__ << "]"
      << " Unexpected FEC crate ("
      << conn.fecCrate() << ") for this module ("
      << key_.fecCrate() << ")!";
    return;
  }

  if ( key_.fecSlot() && key_.fecSlot() != conn.fecSlot() ) {
    edm::LogWarning(mlCabling_)
            << "SiStripModule::" << __func__ << "]"
	    << " Unexpected FEC slot ("
	    << conn.fecSlot() << ") for this module ("
	    << key_.fecSlot() << ")!";
    return;
  }

  if ( key_.fecRing() && key_.fecRing() != conn.fecRing() ) {
    edm::LogWarning(mlCabling_)
      << "SiStripModule::" << __func__ << "]"
      << " Unexpected FEC ring ("
      << conn.fecRing() << ") for this module ("
      << key_.fecRing() << ")!";
    return;
  }

  if ( key_.ccuAddr() && key_.ccuAddr() != conn.ccuAddr() ) {
    edm::LogWarning(mlCabling_)
      << "SiStripModule::" << __func__ << "]"
      << " Unexpected CCU addr ("
      << conn.ccuAddr() << ") for this module ("
      << key_.ccuAddr() << ")!";
    return;
  }

  if ( key_.ccuChan() && key_.ccuChan() != conn.ccuChan() ) {
    edm::LogWarning(mlCabling_)
      << "SiStripModule::" << __func__ << "]"
      << " Unexpected CCU chan ("
      << conn.ccuChan() << ") for this module ("
      << key_.ccuChan() << ")!";
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
  FedChannel fed_ch( conn.fedCrate(), 
		     conn.fedSlot(), 
		     conn.fedId(), 
		     conn.fedCh() ); 
  fedCh( conn.i2cAddr(0), fed_ch );
  
  // DCU, MUX, PLL, LLD
  if ( conn.dcu() ) { dcu0x00_ = true; }
  if ( conn.mux() ) { mux0x43_ = true; }
  if ( conn.pll() ) { pll0x44_ = true; }
  if ( conn.lld() ) { lld0x60_ = true; }
  
}

// -----------------------------------------------------------------------------
//
std::vector<uint16_t> SiStripModule::activeApvs() const {
  std::vector<uint16_t> apvs;
  if ( apv32_ ) { apvs.push_back( apv32_ ); }
  if ( apv33_ ) { apvs.push_back( apv33_ ); }
  if ( apv34_ ) { apvs.push_back( apv34_ ); }
  if ( apv35_ ) { apvs.push_back( apv35_ ); }
  if ( apv36_ ) { apvs.push_back( apv36_ ); }
  if ( apv37_ ) { apvs.push_back( apv37_ ); }
  return apvs;
}

// -----------------------------------------------------------------------------
//
 const uint16_t& SiStripModule::activeApv( const uint16_t& apv_address ) const {
  if      ( apv_address == 0 || apv_address == 32 ) { return apv32_; }
  else if ( apv_address == 1 || apv_address == 33 ) { return apv33_; }
  else if ( apv_address == 2 || apv_address == 34 ) { return apv34_; }
  else if ( apv_address == 3 || apv_address == 35 ) { return apv35_; }
  else if ( apv_address == 4 || apv_address == 36 ) { return apv36_; }
  else if ( apv_address == 5 || apv_address == 37 ) { return apv37_; }
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
  if      ( !apv32_ && apv_address == 32 ) { apv32_ = 32; added_apv = true; }
  else if ( !apv33_ && apv_address == 33 ) { apv33_ = 33; added_apv = true; }
  else if ( !apv34_ && apv_address == 34 ) { apv34_ = 34; added_apv = true; }
  else if ( !apv35_ && apv_address == 35 ) { apv35_ = 35; added_apv = true; }
  else if ( !apv36_ && apv_address == 36 ) { apv36_ = 36; added_apv = true; }
  else if ( !apv37_ && apv_address == 37 ) { apv37_ = 37; added_apv = true; }
  
  std::stringstream ss;
  ss << "SiStripModule::" << __func__ << "]";
  if ( added_apv ) { ss << " Added new APV for"; }
  else { ss << " APV already exists for"; }
  ss << " Crate/FEC/Ring/CCU/Module: "
     << key_.fecCrate() << "/"
     << key_.fecSlot() << "/"
     << key_.fecRing() << "/"
     << key_.ccuAddr() << "/"
     << key_.ccuChan() << "/"
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
    if ( apv32_ || apv33_ ) { nApvPairs_++; }
    if ( apv34_ || apv35_ ) { nApvPairs_++; }
    if ( apv36_ || apv37_ ) { nApvPairs_++; }
  } else { 
    edm::LogWarning(mlCabling_)
      << "SiStripModule::" << __func__ << "]"
      << " Unexpected number of APV pairs: " 
      << npairs;
  }
} 

// -----------------------------------------------------------------------------
//
SiStripModule::PairOfU16 SiStripModule::activeApvPair( const uint16_t& lld_channel ) const {
  if      ( lld_channel == 1 ) { return PairOfU16(apv32_,apv33_); }
  else if ( lld_channel == 2 ) { return PairOfU16(apv34_,apv35_); }
  else if ( lld_channel == 3 ) { return PairOfU16(apv36_,apv37_); }
  else { 
    edm::LogWarning(mlCabling_)
      << "SiStripModule::" << __func__ << "]"
      << " Unexpected LLD channel: " << lld_channel;
    return PairOfU16(0,0); 
  }
}

// -----------------------------------------------------------------------------
//
uint16_t SiStripModule::lldChannel( const uint16_t& apv_pair_num ) const {
  if ( apv_pair_num > 2 ) {
    edm::LogWarning(mlCabling_)
      << "SiStripModule::" << __func__ << "]"
      << " Unexpected APV pair number: " << apv_pair_num;
    return 0;
  }
  if ( nApvPairs_ != 2 && nApvPairs_ != 3 ) {
    edm::LogWarning(mlCabling_)
      << "SiStripModule::" << __func__ << "]"
      << " Unexpected number of APV pairs: " << nApvPairs_;
    return 0;
  }
  if ( nApvPairs_ == 2 && apv_pair_num == 1 ) { return 3; }
  else if ( nApvPairs_ == 2 && apv_pair_num == 2 ) { 
    edm::LogWarning(mlCabling_)
      << "[SiStripFecCabling::" << __func__ << "]"
      << " APV pair number is incompatible with"
      << " respect to number of !";
    return 0;
  } else { return apv_pair_num + 1; } 
}

// -----------------------------------------------------------------------------
//
uint16_t SiStripModule::apvPairNumber( const uint16_t& lld_channel ) const {
  if ( lld_channel < 1 || lld_channel > 3 ) {
    edm::LogWarning(mlCabling_)
      << "SiStripModule::" << __func__ << "]"
      << " Unexpected LLD channel: " << lld_channel;
    return 0;
  }
  if ( nApvPairs_ != 2 && nApvPairs_ != 3 ) {
    edm::LogWarning(mlCabling_)
      << "SiStripModule::" << __func__ << "]"
      << " Unexpected number of APV pairs: " << nApvPairs_;
    return 0;
  }
  if ( nApvPairs_ == 2 && lld_channel == 3 ) { return 1; }
  else if ( nApvPairs_ == 2 && lld_channel == 2 ) { 
    edm::LogWarning(mlCabling_)
      << "SiStripModule::" << __func__ << "]"
      << " LLD channel is incompatible with"
      << " respect to number of APV pairs!";
    return 0;
  } else { return lld_channel - 1; }
}

// -----------------------------------------------------------------------------
//
SiStripModule::FedChannel SiStripModule::fedCh( const uint16_t& apv_pair ) const {

  FedChannel fed_ch(0,0,0,0);
  
  if ( !nApvPairs() ) {
    
    edm::LogWarning(mlCabling_)
      << "SiStripModule::" << __func__ << "]"
      << " No APV pairs exist!";
    return fed_ch; 

  } else {

    uint16_t lld_ch = 0;
    if ( nApvPairs() == 2 ) {

      if      ( apv_pair == 0 ) { lld_ch = 1; }
      else if ( apv_pair == 1 ) { lld_ch = 3; }
      else { 
	edm::LogWarning(mlCabling_)
	  << "SiStripModule::" << __func__ << "]"
	  << " Unexpected pair number! " << apv_pair;
      }

    } else if ( nApvPairs() == 3 ) {

      if      ( apv_pair == 0 ) { lld_ch = 1; }
      else if ( apv_pair == 1 ) { lld_ch = 2; }
      else if ( apv_pair == 2 ) { lld_ch = 3; }
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
  int16_t lld_ch = 1;
  if      ( apv_address == 32 || apv_address == 33 ) { lld_ch = 1; }
  else if ( apv_address == 34 || apv_address == 35 ) { lld_ch = 2; }
  else if ( apv_address == 36 || apv_address == 37 ) { lld_ch = 3; }
  else if ( apv_address == 0 ) { ; } //@@ do nothing?
  else { 
    edm::LogWarning(mlCabling_) << "[SiStripModule::fedCh]" 
				<< " Unexpected I2C address (" 
				<< apv_address << ") for APV!"; 
    return false;
  }
  // Search for entry in std::map
  //@@ use FedKey as key instead of lld chan? what about "duplicates"? 
  //@@ always append to std::map? then can have >3 entries. useful for debug?
  FedCabling::iterator ipair = cabling_.find( lld_ch );
  if ( ipair == cabling_.end() ) { cabling_[lld_ch] = fed_ch; }
  else { ipair->second = fed_ch; }
  return true;
}

// -----------------------------------------------------------------------------
//
void SiStripModule::print( std::stringstream& ss ) const {

  ss << " [SiStripModule::" << __func__ << "]" << std::endl
     << " Crate/FEC/Ring/CCU/Module               : "
     << key().fecCrate() << "/"
     << key().fecSlot() << "/"
     << key().fecRing() << "/"
     << key().ccuAddr() << "/"
     << key().ccuChan() << std::endl;

  ss << " ActiveApvs                              : ";
  std::vector<uint16_t> apvs = activeApvs();
  if ( apvs.empty() ) { ss << "NONE!"; }
  std::vector<uint16_t>::const_iterator iapv = apvs.begin();
  for ( ; iapv != apvs.end(); iapv++ ) { ss << *iapv << ", "; }
  ss << std::endl;
  
  ss << " DcuId/DetId/nPairs                      : "
     << std::hex
     << "0x" << std::setfill('0') << std::setw(8) << dcuId() << "/"
     << "0x" << std::setfill('0') << std::setw(8) << detId() << "/"
     << std::dec
     << nApvPairs() << std::endl;
  
  FedCabling channels = fedChannels();
  ss << " ApvPairNum/FedCrate/FedSlot/FedId/FedCh : ";
  FedCabling::const_iterator ichan = channels.begin();
  for ( ; ichan != channels.end(); ichan++ ) {
    ss << ichan->first << "/"
       << ichan->second.fedCrate_ << "/"
       << ichan->second.fedSlot_ << "/"
       << ichan->second.fedId_ << "/"
       << ichan->second.fedCh_ << ", ";
  }
  ss << std::endl;
  
  ss << " DCU/MUX/PLL/LLD found                   : "
     << bool(dcu0x00_) << "/"
     << bool(mux0x43_) << "/"
     << bool(pll0x44_) << "/"
     << bool(lld0x60_);
  
}

// -----------------------------------------------------------------------------
//@@ NEEDS MODIFYING!!!!
void SiStripModule::terse( std::stringstream& ss ) const {

  ss << " [SiStripModule::" << __func__ << "]" << std::endl
     << " Crate/FEC/Ring/CCU/Module               : "
     << key().fecCrate() << "/"
     << key().fecSlot() << "/"
     << key().fecRing() << "/"
     << key().ccuAddr() << "/"
     << key().ccuChan() << std::endl;

  ss << " ActiveApvs                              : ";
  std::vector<uint16_t> apvs = activeApvs();
  if ( apvs.empty() ) { ss << "NONE!"; }
  std::vector<uint16_t>::const_iterator iapv = apvs.begin();
  for ( ; iapv != apvs.end(); iapv++ ) { ss << *iapv << ", "; }
  ss << std::endl;
  
  ss << " DcuId/DetId/nPairs                      : "
     << std::hex
     << "0x" << std::setfill('0') << std::setw(8) << dcuId() << "/"
     << "0x" << std::setfill('0') << std::setw(8) << detId() << "/"
     << std::dec
     << nApvPairs() << std::endl;
  
  FedCabling channels = fedChannels();
  ss << " ApvPairNum/FedCrate/FedSlot/FedId/FedCh : ";
  FedCabling::const_iterator ichan = channels.begin();
  for ( ; ichan != channels.end(); ichan++ ) {
    ss << ichan->first << "/"
       << ichan->second.fedCrate_ << "/"
       << ichan->second.fedSlot_ << "/"
       << ichan->second.fedId_ << "/"
       << ichan->second.fedCh_ << ", ";
  }
  ss << std::endl;
  
  ss << " DCU/MUX/PLL/LLD found                   : "
     << bool(dcu0x00_) << "/"
     << bool(mux0x43_) << "/"
     << bool(pll0x44_) << "/"
     << bool(lld0x60_);
  
}

// -----------------------------------------------------------------------------
//
std::ostream& operator<< ( std::ostream& os, const SiStripModule& device ) {
  std::stringstream ss;
  device.print(ss);
  os << ss.str();
  return os;
}
