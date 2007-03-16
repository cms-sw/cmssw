#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>
#include <sstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripFecCabling::SiStripFecCabling( const SiStripFedCabling& fed_cabling ) : crates_() {
  LogTrace(mlCabling_)
    << "[SiStripFecCabling::" << __func__ << "]"
    << " Constructing object...";
  buildFecCabling( fed_cabling );
}

// -----------------------------------------------------------------------------
//
void SiStripFecCabling::buildFecCabling( const SiStripFedCabling& fed_cabling ) {
  LogTrace(mlCabling_)
    << "[SiStripFecCabling::" << __func__ << "]"
    << " Building FEC cabling...";
  // Retrieve and iterate through FED ids
  const std::vector<uint16_t>& feds = fed_cabling.feds();
  std::vector<uint16_t>::const_iterator ifed;
  for ( ifed = feds.begin(); ifed != feds.end(); ifed++ ) {
    // Retrieve and iterate through FED channel connections
    const std::vector<FedChannelConnection>& conns = fed_cabling.connections( *ifed ); 
    std::vector<FedChannelConnection>::const_iterator iconn;
    for ( iconn = conns.begin(); iconn != conns.end(); iconn++ ) {
      // Check that FED id is non-zero and add devices
      if ( iconn->fedId() ) { addDevices( *iconn ); } 
    }
  }
  // Consistency checks
  for ( std::vector<SiStripFecCrate>::const_iterator icrate = this->crates().begin(); icrate != this->crates().end(); icrate++ ) {
    for ( std::vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++ ) {
      for ( std::vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
	for ( std::vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
	  for ( std::vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); imod != iccu->modules().end(); imod++ ) {
	    //@@ need consistency checks here!
	  }
	}
      }
    }
  }
  
}

// -----------------------------------------------------------------------------
//
void SiStripFecCabling::addDevices( const FedChannelConnection& conn ) {
  std::vector<SiStripFecCrate>::const_iterator icrate = crates().begin();
  while ( icrate != crates().end() && (*icrate).fecCrate() != conn.fecCrate() ) { icrate++; }
  if ( icrate == crates().end() ) { 
    crates_.push_back( SiStripFecCrate( conn ) ); 
  } else { 
    const_cast<SiStripFecCrate&>(*icrate).addDevices( conn ); 
  }
}

// -----------------------------------------------------------------------------
//
void SiStripFecCabling::connections( std::vector<FedChannelConnection>& conns ) const {
  conns.clear();
  for ( std::vector<SiStripFecCrate>::const_iterator icrate = this->crates().begin(); icrate != this->crates().end(); icrate++ ) {
    for ( std::vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++ ) {
      for ( std::vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
	for ( std::vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
	  for ( std::vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); imod != iccu->modules().end(); imod++ ) {
	    for ( uint16_t ipair = 0; ipair < imod->nApvPairs(); ipair++ ) {
	      conns.push_back( FedChannelConnection( icrate->fecCrate(), ifec->fecSlot(), iring->fecRing(), iccu->ccuAddr(), imod->ccuChan(), 
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
}

// -----------------------------------------------------------------------------
//
const SiStripModule& SiStripFecCabling::module( const FedChannelConnection& conn ) const {
  std::vector<SiStripFecCrate>::const_iterator icrate = crates().begin();
  while ( icrate != crates().end() && icrate->fecCrate() != conn.fecCrate() ) { icrate++; }
  if ( icrate != crates().end() ) { 
    std::vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin();
    while ( ifec != icrate->fecs().end() && ifec->fecSlot() != conn.fecSlot() ) { ifec++; }
    if ( ifec != icrate->fecs().end() ) { 
      std::vector<SiStripRing>::const_iterator iring = ifec->rings().begin();
      while ( iring != ifec->rings().end() && iring->fecRing() != conn.fecRing() ) { iring++; }
      if ( iring != ifec->rings().end() ) { 
	std::vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin();
	while ( iccu != iring->ccus().end() && iccu->ccuAddr() != conn.ccuAddr() ) { iccu++; }
	if ( iccu != iring->ccus().end() ) { 
	  std::vector<SiStripModule>::const_iterator imod = iccu->modules().begin();
	  while ( imod != iccu->modules().end() && imod->ccuChan() != conn.ccuChan() ) { imod++; }
	  if ( imod != iccu->modules().end() ) { 
	    return *imod;
	  } else { edm::LogWarning(mlCabling_)
	    << "[SiStripFecCabling::" << __func__ << "]"
	    << " CCU channel " << conn.ccuChan() 
	    << " not found!"; }
	} else { edm::LogWarning(mlCabling_)
	  << "[SiStripFecCabling::" << __func__ << "]"
	  << " CCU address " << conn.ccuAddr() 
	  << " not found!"; }
      } else { edm::LogWarning(mlCabling_)
	<< "[SiStripFecCabling::" << __func__ << "]"
	<< " FEC ring " << conn.fecRing() 
	<< " not found!"; }
    } else { edm::LogWarning(mlCabling_)
      << "[SiStripFecCabling::" << __func__ << "]"
      << " FEC slot " << conn.fecSlot() 
      << " not found!"; }
  } else { edm::LogWarning(mlCabling_)
    << "[SiStripFecCabling::" << __func__ << "]"
    << " FEC crate " << conn.fecCrate() 
    << " not found!"; }
  static FedChannelConnection temp;
  static const SiStripModule module(temp);
  return module;
}

// -----------------------------------------------------------------------------
//
const SiStripModule& SiStripFecCabling::module( const uint32_t& dcu_id ) const {
  for ( std::vector<SiStripFecCrate>::const_iterator icrate = this->crates().begin(); icrate != this->crates().end(); icrate++ ) {
    for ( std::vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++ ) {
      for ( std::vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
	for ( std::vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
	  for ( std::vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); imod != iccu->modules().end(); imod++ ) {
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

  std::vector<uint16_t> fed_ids; std::vector<uint16_t>::iterator ifed;
  for ( std::vector<SiStripFecCrate>::const_iterator icrate = this->crates().begin(); icrate != this->crates().end(); icrate++ ) {
    for ( std::vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++ ) {
      for ( std::vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
	for ( std::vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
	  for ( std::vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); imod != iccu->modules().end(); imod++ ) {

	    // APVs
	    if ( imod->activeApv(32) ) { num_of_devices.nApvs_++; }
	    if ( imod->activeApv(33) ) { num_of_devices.nApvs_++; }
	    if ( imod->activeApv(34) ) { num_of_devices.nApvs_++; }
	    if ( imod->activeApv(35) ) { num_of_devices.nApvs_++; }
	    if ( imod->activeApv(36) ) { num_of_devices.nApvs_++; }
	    if ( imod->activeApv(37) ) { num_of_devices.nApvs_++; }
	    if ( imod->dcuId() ) { num_of_devices.nDcuIds_++; }
	    if ( imod->detId() ) { num_of_devices.nDetIds_++; }

	    // APV pairs
	    num_of_devices.nApvPairs_ += imod->nApvPairs();
	    if      ( imod->nApvPairs() == 0 ) { num_of_devices.nApvPairs0_++; }
	    else if ( imod->nApvPairs() == 1 ) { num_of_devices.nApvPairs1_++; }
	    else if ( imod->nApvPairs() == 2 ) { num_of_devices.nApvPairs2_++; }
	    else if ( imod->nApvPairs() == 3 ) { num_of_devices.nApvPairs3_++; }
	    else { num_of_devices.nApvPairsX_++; }

	    // FED ids and channels
	    for ( uint16_t ipair = 0; ipair < imod->nApvPairs(); ipair++ ) {
	      uint16_t fed_id = imod->fedCh(ipair).first;
	      if ( fed_id ) { 
		ifed = find ( fed_ids.begin(), fed_ids.end(), fed_id );
		if ( ifed != fed_ids.end() ) { num_of_devices.nFedIds_++; }
		num_of_devices.nFedChans_++;
	      }
	    }

	    // FE devices
	    if ( imod->dcu() ) { num_of_devices.nDcus_++; }
	    if ( imod->mux() ) { num_of_devices.nMuxes_++; }
	    if ( imod->pll() ) { num_of_devices.nPlls_++; }
	    if ( imod->lld() ) { num_of_devices.nLlds_++; }

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
  
  return num_of_devices;
  
}

// -----------------------------------------------------------------------------
// 
EVENTSETUP_DATA_REG(SiStripFecCabling);
