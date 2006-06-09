#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>
#include <sstream>

using namespace std;

// -----------------------------------------------------------------------------
//
SiStripFecCabling::SiStripFecCabling( const SiStripFedCabling& fed_cabling ) : crates_() {
  edm::LogInfo("FecCabling") << "[SiStripFecCabling::SiStripFecCabling] Constructing object...";
  buildFecCabling( fed_cabling );
}

// -----------------------------------------------------------------------------
//
void SiStripFecCabling::buildFecCabling( const SiStripFedCabling& fed_cabling ) {
  edm::LogInfo("FecCabling") << "[SiStripFecCabling::buildFecCabling]";
  // Retrieve and iterate through FED ids
  const vector<uint16_t>& feds = fed_cabling.feds();
  vector<uint16_t>::const_iterator ifed;
  for ( ifed = feds.begin(); ifed != feds.end(); ifed++ ) {
    // Retrieve and iterate through FED channel connections
    const vector<FedChannelConnection>& conns = fed_cabling.connections( *ifed ); 
    vector<FedChannelConnection>::const_iterator iconn;
    for ( iconn = conns.begin(); iconn != conns.end(); iconn++ ) {
      // Check that FED id is non-zero and add devices
      if ( iconn->fedId() ) { addDevices( *iconn ); } 
    }
  }
  // Consistency checks
  for ( vector<SiStripFecCrate>::const_iterator icrate = this->crates().begin(); icrate != this->crates().end(); icrate++ ) {
    for ( vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++ ) {
      for ( vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
	for ( vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
	  for ( vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); imod != iccu->modules().end(); imod++ ) {
	    imod->print(); //@@ need consistency checks here!
	  }
	}
      }
    }
  }
  
}

// -----------------------------------------------------------------------------
//
void SiStripFecCabling::addDevices( const FedChannelConnection& conn ) {
  //   LogDebug("FecCabling") << "[SiStripFecCabling::addDevices]" 
  // 			 << " Adding new Device with following I2C addresses. " 
  // 			 << " FEC crate: " << conn.fecCrate()
  // 			 << " FEC slot: " << conn.fecSlot()
  // 			 << " FEC ring: " << conn.fecRing()
  // 			 << " CCU addr: " << conn.ccuAddr()
  // 			 << " CCU chan: " << conn.ccuChan();
  vector<SiStripFecCrate>::const_iterator icrate = crates().begin();
  while ( icrate != crates().end() && (*icrate).fecCrate() != conn.fecCrate() ) { icrate++; }
  if ( icrate == crates().end() ) { 
    //     LogDebug("FecCabling") << "[SiStripFecCabling::addDevices]" 
    // 			   << " Adding new FEC crate with address " 
    // 			   << conn.fecCrate();
    crates_.push_back( SiStripFecCrate( conn ) ); 
  } else { 
    //     LogDebug("FecCabling") << "[SiStripFecCabling::addDevices]" 
    // 			   << " FEC crate already exists with address " 
    // 			   << icrate->fecCrate();
    const_cast<SiStripFecCrate&>(*icrate).addDevices( conn ); 
  }
}

// -----------------------------------------------------------------------------
//
void SiStripFecCabling::connections( vector<FedChannelConnection>& conns ) const {
  conns.clear();
  for ( vector<SiStripFecCrate>::const_iterator icrate = this->crates().begin(); icrate != this->crates().end(); icrate++ ) {
    for ( vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++ ) {
      for ( vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
	for ( vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
	  for ( vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); imod != iccu->modules().end(); imod++ ) {
	    for ( uint16_t ipair = 0; ipair < imod->nApvPairs(); ipair++ ) {
	      conns.push_back( FedChannelConnection( 0, ifec->fecSlot(), iring->fecRing(), iccu->ccuAddr(), imod->ccuChan(), 
						     imod->activeApvPair( imod->lldChannel(ipair) ).first, 
						     imod->activeApvPair( imod->lldChannel(ipair) ).second,
						     imod->dcuId(), imod->detId(), imod->nApvPairs(),
						     imod->fedCh(ipair).first, imod->fedCh(ipair).second, 0, //imod->length(),
						     imod->dcu(), imod->pll(), imod->mux(), imod->lld() ) );
	    }
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
  vector<SiStripFecCrate>::const_iterator icrate = crates().begin();
  while ( icrate != crates().end() && icrate->fecCrate() != conn.fecCrate() ) { icrate++; }
  if ( icrate != crates().end() ) { 
    vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin();
    while ( ifec != icrate->fecs().end() && ifec->fecSlot() != conn.fecSlot() ) { ifec++; }
    if ( ifec != icrate->fecs().end() ) { 
      vector<SiStripRing>::const_iterator iring = ifec->rings().begin();
      while ( iring != ifec->rings().end() && iring->fecRing() != conn.fecRing() ) { iring++; }
      if ( iring != ifec->rings().end() ) { 
	vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin();
	while ( iccu != iring->ccus().end() && iccu->ccuAddr() != conn.ccuAddr() ) { iccu++; }
	if ( iccu != iring->ccus().end() ) { 
	  vector<SiStripModule>::const_iterator imod = iccu->modules().begin();
	  while ( imod != iccu->modules().end() && imod->ccuChan() != conn.ccuChan() ) { imod++; }
	  if ( imod != iccu->modules().end() ) { 
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
  } else { edm::LogError("FecCabling") << "[SiStripFecCabling::module]"
				       << " FEC crate " << conn.fecCrate() 
				       << " not found!"; }
  static FedChannelConnection temp;
  static const SiStripModule module(temp);
  return module;
}

// -----------------------------------------------------------------------------
//
const SiStripModule& SiStripFecCabling::module( const uint32_t& dcu_id ) const {
  for ( vector<SiStripFecCrate>::const_iterator icrate = this->crates().begin(); icrate != this->crates().end(); icrate++ ) {
    for ( vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++ ) {
      for ( vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
	for ( vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
	  for ( vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); imod != iccu->modules().end(); imod++ ) {
	    if ( (*imod).dcuId() == dcu_id ) { return *imod; }
	  }
	}
      }
    }
  }
  static FedChannelConnection temp;
  static const SiStripModule module(temp);
  return module;
}

// -----------------------------------------------------------------------------
//
const NumberOfDevices& SiStripFecCabling::countDevices() const {

  static NumberOfDevices num_of_devices; // simple container class used for counting
  num_of_devices.clear();

  vector<uint16_t> fed_ids; vector<uint16_t>::iterator ifed;
  for ( vector<SiStripFecCrate>::const_iterator icrate = this->crates().begin(); icrate != this->crates().end(); icrate++ ) {
    for ( vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++ ) {
      for ( vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
	for ( vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
	  for ( vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); imod != iccu->modules().end(); imod++ ) {
	    // APVs
	    if ( (*imod).activeApv(32) ) { num_of_devices.nApvs_++; }
	    if ( (*imod).activeApv(33) ) { num_of_devices.nApvs_++; }
	    if ( (*imod).activeApv(34) ) { num_of_devices.nApvs_++; }
	    if ( (*imod).activeApv(35) ) { num_of_devices.nApvs_++; }
	    if ( (*imod).activeApv(36) ) { num_of_devices.nApvs_++; }
	    if ( (*imod).activeApv(37) ) { num_of_devices.nApvs_++; }
	    if ( (*imod).dcuId() ) { num_of_devices.nDcuIds_++; }
	    if ( (*imod).detId() ) { num_of_devices.nDetIds_++; }
	    // APV pairs
	    num_of_devices.nApvPairs_ += (*imod).nApvPairs();
	    // FED ids and channels
	    for ( uint16_t ipair = 0; ipair < (*imod).nApvPairs(); ipair++ ) {
	      uint16_t fed_id = (*imod).fedCh(ipair).first;
	      if ( fed_id ) { 
		ifed = find ( fed_ids.begin(), fed_ids.end(), fed_id );
		if ( ifed != fed_ids.end() ) { num_of_devices.nFedIds_++; }
		num_of_devices.nFedChans_++;
	      }
	    }
	    // FE devices
	    if ( (*imod).dcu() ) { num_of_devices.nDcus_++; }
	    if ( (*imod).mux() ) { num_of_devices.nMuxes_++; }
	    if ( (*imod).pll() ) { num_of_devices.nPlls_++; }
	    if ( (*imod).lld() ) { num_of_devices.nLlds_++; }
	    // FE modules
	    num_of_devices.nCcuChans_++;
	  } 
	  num_of_devices.nCcuAddrs_++;
	}
	num_of_devices.nFecRings_++;
      }
      num_of_devices.nFecSlots_++;
    }
    num_of_devices.nFecCrates_++;
  }
  
  stringstream ss; 
  num_of_devices.print( ss );
  LogDebug("FecCabling") << "[SiStripFecCabling::countDevices]" << ss.str();
  return num_of_devices;
  
}

// -----------------------------------------------------------------------------
// 
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
EVENTSETUP_DATA_REG(SiStripFecCabling);

// -----------------------------------------------------------------------------
//
void NumberOfDevices::clear() {
  nFecCrates_ = 0;
  nFecSlots_ = 0;
  nFecRings_ = 0;
  nCcuAddrs_ = 0;
  nCcuChans_ = 0;
  nApvs_ = 0;
  nDcuIds_ = 0;
  nDetIds_ = 0;
  nApvPairs_ = 0;
  nFedIds_ = 0;
  nFedChans_ = 0;
  nDcus_ = 0;
  nMuxes_ = 0;
  nPlls_ = 0;
  nLlds_ = 0;
}

// -----------------------------------------------------------------------------
//
void NumberOfDevices::print( stringstream& ss ) {
  ss.str("");
  ss << "[NumberOfDevices::print] Number of devices found: " 
     << "  FEC crates: " << nFecCrates_
     << "  FEC slots: " << nFecSlots_
     << "  FEC rings: " << nFecRings_
     << "  CCU addrs: " << nCcuAddrs_
     << "  CCU chans: " << nCcuChans_
     << "  APVs: " << nApvs_
     << "  DCU ids: " << nDcuIds_
     << "  DET ids: " << nDetIds_
     << "  APV pairs: " << nApvPairs_
     << "  FED channels: " << nFedChans_
     << "  DCUs: " << nDcus_
     << "  MUXes: " << nMuxes_
     << "  PLLs: " << nPlls_
     << "  LLDs: " << nLlds_;
}





