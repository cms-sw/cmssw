#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>
#include <sstream>

// -----------------------------------------------------------------------------
//
SiStripFecCabling::SiStripFecCabling( const SiStripFedCabling& cabling ) : fecs_() {
  edm::LogInfo("FecCabling") << "[SiStripFecCabling::SiStripFecCabling] Constructing object...";
  const vector<uint16_t>& feds = cabling.feds();
  vector<uint16_t>::const_iterator ifed;
  for ( ifed = feds.begin(); ifed != feds.end(); ifed++ ) {
    const vector<FedChannelConnection>& conns = cabling.connections( *ifed ); 
    vector<FedChannelConnection>::const_iterator iconn;
    for ( iconn = conns.begin(); iconn != conns.end(); iconn++ ) {
      addDevices( *iconn );
    }
  }

  for ( vector<SiStripFec>::const_iterator ifec = this->fecs().begin(); ifec != this->fecs().end(); ifec++ ) {
    for ( vector<SiStripRing>::const_iterator iring = (*ifec).rings().begin(); iring != (*ifec).rings().end(); iring++ ) {
      for ( vector<SiStripCcu>::const_iterator iccu = (*iring).ccus().begin(); iccu != (*iring).ccus().end(); iccu++ ) {
	for ( vector<SiStripModule>::const_iterator imodule = (*iccu).modules().begin(); imodule != (*iccu).modules().end(); imodule++ ) {
	  imodule->print();
	}
      }
    }
  }
  
  
}

// -----------------------------------------------------------------------------
//
void SiStripFecCabling::addDevices( const FedChannelConnection& conn ) {
  LogDebug("FecCabling") << "[SiStripFecCabling::addDevices]" 
			 << " Adding new Device with following I2C addresses. " 
			 << " FEC slot: " << conn.fecSlot()
			 << " FEC ring: " << conn.fecRing()
			 << " CCU addr: " << conn.ccuAddr()
			 << " CCU chan: " << conn.ccuChan();
  vector<SiStripFec>::const_iterator ifec = fecs().begin();
  while ( ifec != fecs().end() && (*ifec).fecSlot() != conn.fecSlot() ) { ifec++; }
  if ( ifec == fecs().end() ) { 
    LogDebug("FecCabling") << "[SiStripFecCabling::addDevices]" 
			   << " Adding new FEC with address " 
			   << conn.fecSlot();
    fecs_.push_back( SiStripFec( conn ) ); 
  } else { 
    LogDebug("FecCabling") << "[SiStripFecCabling::addDevices]" 
			   << " FEC already exists with address " 
			   << ifec->fecSlot();
    const_cast<SiStripFec&>(*ifec).addDevices( conn ); 
  }
}

// -----------------------------------------------------------------------------
//
void SiStripFecCabling::connections( vector<FedChannelConnection>& conns ) {
  conns.clear();
  for ( vector<SiStripFec>::const_iterator ifec = (*this).fecs().begin(); ifec != (*this).fecs().end(); ifec++ ) {
    for ( vector<SiStripRing>::const_iterator iring = (*ifec).rings().begin(); iring != (*ifec).rings().end(); iring++ ) {
      for ( vector<SiStripCcu>::const_iterator iccu = (*iring).ccus().begin(); iccu != (*iring).ccus().end(); iccu++ ) {
	for ( vector<SiStripModule>::const_iterator imod = (*iccu).modules().begin(); imod != (*iccu).modules().end(); imod++ ) {
	  for ( uint16_t ipair = 0; ipair < (*imod).nApvPairs(); ipair++ ) {
	    conns.push_back( FedChannelConnection( 0, (*ifec).fecSlot(), (*iring).fecRing(), (*iccu).ccuAddr(), (*imod).ccuChan(), 
						   (*imod).activeApvPair(ipair).first, (*imod).activeApvPair(ipair).second,
						   (*imod).dcuId(), (*imod).detId(), (*imod).nApvPairs(),
						   (*imod).fedCh(ipair).first, (*imod).fedCh(ipair).second, 0, //(*imod).length(),
						   (*imod).dcu(), (*imod).pll(), (*imod).mux(), (*imod).lld() ) );
	  }
	}
      }
    }
  }
  edm::LogInfo("FecCabling") << "[SiStripFecCabling::connections]" 
			     << " Found " << conns.size() << " FED channel connection objects";
}

// -----------------------------------------------------------------------------
//
const SiStripModule& SiStripFecCabling::module( const FedChannelConnection& conn ) const {
  vector<SiStripFec>::const_iterator ifec = fecs().begin();
  while ( ifec != fecs().end() && (*ifec).fecSlot() != conn.fecSlot() ) { ifec++; }
  if ( ifec != fecs().end() ) { 
    vector<SiStripRing>::const_iterator iring = (*ifec).rings().begin();
    while ( iring != (*ifec).rings().end() && (*iring).fecRing() != conn.fecRing() ) { iring++; }
    if ( iring != (*ifec).rings().end() ) { 
      vector<SiStripCcu>::const_iterator iccu = (*iring).ccus().begin();
      while ( iccu != (*iring).ccus().end() && (*iccu).ccuAddr() != conn.ccuAddr() ) { iccu++; }
      if ( iccu != (*iring).ccus().end() ) { 
	vector<SiStripModule>::const_iterator imod = (*iccu).modules().begin();
	while ( imod != (*iccu).modules().end() && (*imod).ccuChan() != conn.ccuChan() ) { imod++; }
	if ( imod != (*iccu).modules().end() ) { 
	  return *imod;
	} else { edm::LogError("FecCabling") << "[SiStripFecCabling::module]"
					     << " CCU channel " << conn.ccuChan() 
					     << " not found!"; }
      } else { edm::LogError("FecCabling") << "[SiStripFecCabling::module]"
					   << " CCU address " << conn.ccuAddr() 
					   << " not found!"; }
    } else { edm::LogError("FecCabling") << "[SiStripFecCabling::module]"
					 << " FEC ring " << conn.fecRing() 
					 << " not found!"; }
  } else { edm::LogError("FecCabling") << "[SiStripFecCabling::module]"
				       << " FEC slot " << conn.fecSlot() 
				       << " not found!"; }
  static FedChannelConnection temp;
  static const SiStripModule module(temp);
  return module;
}

// -----------------------------------------------------------------------------
//
void SiStripFecCabling::countDevices() {
  uint32_t nfecs, nrings, nccus, nmodules, napvs, ndcuids, ndetids;
  uint32_t npairs, nfedchans, ndcus, nmuxes, nplls, nllds;
  nfecs = nrings = nccus = nmodules = napvs = ndcuids = ndetids = 0;
  npairs = nfedchans = ndcus = nmuxes = nplls = nllds = 0;
  for ( vector<SiStripFec>::const_iterator ifec = (*this).fecs().begin(); ifec != (*this).fecs().end(); ifec++ ) {
    for ( vector<SiStripRing>::const_iterator iring = (*ifec).rings().begin(); iring != (*ifec).rings().end(); iring++ ) {
      for ( vector<SiStripCcu>::const_iterator iccu = (*iring).ccus().begin(); iccu != (*iring).ccus().end(); iccu++ ) {
	for ( vector<SiStripModule>::const_iterator imod = (*iccu).modules().begin(); imod != (*iccu).modules().end(); imod++ ) {
	  // APVs
	  if ( (*imod).activeApv(32) ) { napvs++; }
	  if ( (*imod).activeApv(33) ) { napvs++; }
	  if ( (*imod).activeApv(34) ) { napvs++; }
	  if ( (*imod).activeApv(35) ) { napvs++; }
	  if ( (*imod).activeApv(36) ) { napvs++; }
	  if ( (*imod).activeApv(37) ) { napvs++; }
	  if ( (*imod).dcuId() ) { ndcuids++; }
	  if ( (*imod).detId() ) { ndetids++; }
	  // APV pairs
	  npairs += (*imod).nApvPairs();
	  // FED channels
	  for ( uint16_t ipair = 0; ipair < (*imod).nApvPairs(); ipair++ ) {
	    if ( (*imod).fedCh(ipair).first ) { nfedchans++; }
	  }
	  // FE devices
	  if ( (*imod).dcu() ) { ndcus++; }
	  if ( (*imod).mux() ) { nmuxes++; }
	  if ( (*imod).pll() ) { nplls++; }
	  if ( (*imod).lld() ) { nllds++; }
	  // FE modules
	  nmodules++;
	} 
	nccus++;
      }
      nrings++;
    }
    nfecs++;
  }
  LogDebug("FecCabling") << "[SiStripFecCabling::countDevices]"
			 << " Number of devices found." 
			 << " FEC slots: " << nfecs
			 << " FEC rings: " << nrings
			 << " CCU addrs: " << nccus
			 << " CCU chans: " << nmodules
			 << " APVs: " << napvs
			 << " DCU ids: " << ndcuids
			 << " DET ids: " << ndetids
			 << " APV pairs: " << npairs
			 << " FED channels: " << nfedchans
			 << " DCUs: " << ndcus
			 << " MUXes: " << nmuxes
			 << " PLLs: " << nplls
			 << " LLDs: " << nllds;
}

// -----------------------------------------------------------------------------
//
void SiStripFec::addDevices( const FedChannelConnection& conn ) {
  vector<SiStripRing>::const_iterator iring = rings().begin();
  while ( iring != rings().end() && (*iring).fecRing() != conn.fecRing() ) { iring++; }
  if ( iring == rings().end() ) { 
    LogDebug("FecCabling") << "[SiStripFec::addDevices]" 
			   << " Adding new FEC ring with address " 
			   << conn.fecRing();
    rings_.push_back( SiStripRing( conn ) ); 
  } else { 
    LogDebug("FecCabling") << "[SiStripFec::addDevices]" 
			   << " FEC ring already exists with address " 
			   << iring->fecRing();
    const_cast<SiStripRing&>(*iring).addDevices( conn ); 
  }
}

// -----------------------------------------------------------------------------
//
void SiStripRing::addDevices( const FedChannelConnection& conn ) {
  vector<SiStripCcu>::const_iterator iccu = ccus().begin();
  while ( iccu != ccus().end() && (*iccu).ccuAddr() != conn.ccuAddr() ) { iccu++; }
  if ( iccu == ccus().end() ) { 
    LogDebug("FecCabling") << "[SiStripRing::addDevices]" 
			   << " Adding new CCU with address " 
			   << conn.ccuAddr();
    ccus_.push_back( SiStripCcu( conn ) ); 
  } else { 
    LogDebug("FecCabling") << "[SiStripRing::addDevices]" 
			   << " CCU already exists with address " 
			   << iccu->ccuAddr();
    const_cast<SiStripCcu&>(*iccu).addDevices( conn ); 
  }
}

// -----------------------------------------------------------------------------
//
void SiStripCcu::addDevices( const FedChannelConnection& conn ) {
  vector<SiStripModule>::const_iterator imod = modules().begin();
  while ( imod != modules().end() && (*imod).ccuChan() != conn.ccuChan() ) { imod++; }
  if ( imod == modules().end() ) { 
    LogDebug("FecCabling") << "[SiStripCcu::addDevices]" 
			   << " Adding new module with address " 
			   << conn.ccuChan();
    modules_.push_back( SiStripModule( conn ) ); 
  } else { 
    LogDebug("FecCabling") << "[SiStripRing::addDevices]" 
			   << " Module already exists with address " 
			   << imod->ccuChan();
    const_cast<SiStripModule&>(*imod).addDevices( conn ); 
  }
}

// -----------------------------------------------------------------------------
//
void SiStripModule::addDevices( const FedChannelConnection& conn ) {

  // APVs
  addApv( conn.i2cAddrApv0() );
  addApv( conn.i2cAddrApv1() );

  // Detector
  dcuId( conn.dcuId() ); 
  detId( conn.detId() ); 
  nApvPairs( conn.nApvPairs() ); 
  
  // FED cabling
  pair<uint16_t,uint16_t> fed_ch = pair<uint16_t,uint16_t>( conn.fedId(), conn.fedCh() ); 
  fedCh( conn.i2cAddrApv0(), fed_ch );
  
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
  if      ( apv_address == 32 ) { apv0x32_ = 32; }
  else if ( apv_address == 33 ) { apv0x33_ = 33; }
  else if ( apv_address == 34 ) { apv0x34_ = 34; }
  else if ( apv_address == 35 ) { apv0x35_ = 35; }
  else if ( apv_address == 36 ) { apv0x36_ = 36; }
  else if ( apv_address == 37 ) { apv0x37_ = 37; }
  else if ( apv_address == 0 )  { } //@@ nothing?
  else { edm::LogError("FecCabling") << "[SiStripFecCabling::addApv]" 
				     << " Unexpected I2C address (" 
				     << apv_address << ") for APV!"; }
  stringstream ss;
  ss << "[SiStripModule::addApv] Found following APV devices: ";
  for ( uint16_t iapv = 32; iapv < 38; iapv++ ) {
    if ( activeApv(iapv) ) { ss << activeApv(iapv) << ", "; }
  }
  LogDebug("FecCabling") << ss.str();
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
    edm::LogError("FecCabling") << "[SiStripModule::nApvPairs] Unexpected number of APV pairs: " << npairs;
  }
} 

// -----------------------------------------------------------------------------
//
pair<uint16_t,uint16_t> SiStripModule::activeApvPair( const uint16_t& apv_pair_number ) const {
  if ( nApvPairs_ == 2 ) {
    if      ( apv_pair_number == 0 ) { return pair<uint16_t,uint16_t>(apv0x32_,apv0x33_); }
    else if ( apv_pair_number == 1 ) { return pair<uint16_t,uint16_t>(apv0x36_,apv0x37_); }
    else                             { return pair<uint16_t,uint16_t>(0,0); }
  } else if ( nApvPairs_ == 3 ) {
    if      ( apv_pair_number == 0 ) { return pair<uint16_t,uint16_t>(apv0x32_,apv0x33_); }
    else if ( apv_pair_number == 1 ) { return pair<uint16_t,uint16_t>(apv0x34_,apv0x35_); }
    else if ( apv_pair_number == 2 ) { return pair<uint16_t,uint16_t>(apv0x36_,apv0x37_); }
    else                             { return pair<uint16_t,uint16_t>(0,0); }
  } else {
    edm::LogError("FecCabling") << "[SiStripFecCabling::pair] Unexpected number of pairs!";
  }
  return pair<uint16_t,uint16_t>(0,0);
}

// -----------------------------------------------------------------------------
//
const pair<uint16_t,uint16_t>& SiStripModule::fedCh( const uint16_t& apv_pair ) const {
  static const pair<uint16_t,uint16_t> fed_ch = pair<uint16_t,uint16_t>(0,0);
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
    map< uint16_t, pair<uint16_t,uint16_t> >::const_iterator ipair = cabling_.find( lld_ch );
    if ( ipair != cabling_.end() ) { return (*ipair).second; }
    else { return fed_ch; }
  }
}

// -----------------------------------------------------------------------------
//
bool SiStripModule::fedCh( const uint16_t& apv_address, 
			   const pair<uint16_t,uint16_t>& fed_ch ) {
  // Determine LLD channel
  int16_t lld_ch;
  if      ( apv_address == 32 || apv_address == 33 ) { lld_ch = 0; }
  else if ( apv_address == 34 || apv_address == 35 ) { lld_ch = 1; }
  else if ( apv_address == 36 || apv_address == 37 ) { lld_ch = 2; }
  else if ( apv_address == 0 ) { ; } //@@ 
  else { 
    edm::LogError("FecCabling") << "[SiStripModule::fedCh]" 
				<< " Unexpected I2C address (" 
				<< apv_address << ") for APV!"; 
    return false;
  }
  // Search for entry in map
  map< uint16_t, pair<uint16_t,uint16_t> >::iterator ipair = cabling_.find( lld_ch );
  if ( ipair == cabling_.end() ) { cabling_[lld_ch] = fed_ch; }
  else { ipair->second = fed_ch; }
  return true;
}

// -----------------------------------------------------------------------------
//
void SiStripModule::print() const {
  std::stringstream ss;
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
     << std::hex
     << std::setfill('0') << std::setw(8) << dcuId() << "/"
     << std::setfill('0') << std::setw(8) << detId() << "/"
     << std::dec
     << nApvPairs();
  ss << "  nConnected/apvAddr-FedId-FedCh: " 
     << fedChannels().size() << "/";
  map< uint16_t, pair<uint16_t,uint16_t> >::const_iterator iconn;
  for ( iconn = fedChannels().begin(); iconn != fedChannels().end(); iconn++ ) {
    ss << iconn->first << "-"
       << iconn->second.first << "-"
       << iconn->second.second << "/";
  }
  LogDebug("FedCabling") << ss.str();
}







