// Last commit: $Id: SiStripFecCabling.cc,v 1.29 2010/02/20 20:55:05 wmtan Exp $

#include "FWCore/Utilities/interface/typelookup.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripFecCabling::SiStripFecCabling( const SiStripFedCabling& fed_cabling ) 
  : crates_() 
{
  LogTrace(mlCabling_)
    << "[SiStripFecCabling::" << __func__ << "]"
    << " Constructing object...";
  crates_.reserve(4);
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

      // Check that FED id is not invalid and add devices
      if ( iconn->fedId() != sistrip::invalid_ ) { addDevices( *iconn ); } 
      
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
  LogTrace(mlCabling_)
    << "[SiStripFecCabling::" << __func__ << "]"
    << " Finished building FEC cabling";
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
  LogTrace(mlCabling_)
    << "[SiStripFecCabling::" << __func__ << "]"
    << " Building vector of FedChannelConnection objects...";
  conns.clear();
  for ( std::vector<SiStripFecCrate>::const_iterator icrate = this->crates().begin(); icrate != this->crates().end(); icrate++ ) {
    for ( std::vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++ ) {
      for ( std::vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
	for ( std::vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
	  for ( std::vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); imod != iccu->modules().end(); imod++ ) {
	    for ( uint16_t ipair = 0; ipair < imod->nApvPairs(); ipair++ ) {
	      conns.push_back( FedChannelConnection( icrate->fecCrate(), 
						     ifec->fecSlot(), 
						     iring->fecRing(), 
						     iccu->ccuAddr(), 
						     imod->ccuChan(), 
						     imod->activeApvPair( imod->lldChannel(ipair) ).first, 
						     imod->activeApvPair( imod->lldChannel(ipair) ).second,
						     imod->dcuId(), 
						     imod->detId(), 
						     imod->nApvPairs(),
						     imod->fedCh(ipair).fedId_, 
						     imod->fedCh(ipair).fedCh_, 
						     imod->length(),
						     imod->dcu(), 
						     imod->pll(), 
						     imod->mux(), 
						     imod->lld() ) );
	      uint16_t fed_crate = imod->fedCh(ipair).fedCrate_;
	      uint16_t fed_slot = imod->fedCh(ipair).fedSlot_;
	      conns.back().fedCrate( fed_crate );
	      conns.back().fedSlot( fed_slot );
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

  std::stringstream ss;
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
	  } else { 
	    ss << "[SiStripFecCabling::" << __func__ << "]"
	       << " CCU channel " << conn.ccuChan() 
	       << " not found!"; }
	} else { 
	  ss << "[SiStripFecCabling::" << __func__ << "]"
	     << " CCU address " << conn.ccuAddr() 
	     << " not found!"; }
      } else { 
	ss << "[SiStripFecCabling::" << __func__ << "]"
	   << " FEC ring " << conn.fecRing() 
	   << " not found!"; }
    } else { 
      ss << "[SiStripFecCabling::" << __func__ << "]"
	 << " FEC slot " << conn.fecSlot() 
	 << " not found!"; }
  } else { 
    ss << "[SiStripFecCabling::" << __func__ << "]"
       << " FEC crate " << conn.fecCrate() 
       << " not found!"; 
  }

  if ( !ss.str().empty() ) { edm::LogWarning(mlCabling_) << ss.str(); }
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
NumberOfDevices SiStripFecCabling::countDevices() const {
  
  NumberOfDevices num_of_devices; // simple container class used for counting

  std::vector<uint16_t> fed_crates; 
  std::vector<uint16_t> fed_slots; 
  std::vector<uint16_t> fed_ids; 
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

	    // FED crates, slots, ids, channels
	    for ( uint16_t ipair = 0; ipair < imod->nApvPairs(); ipair++ ) {

	      uint16_t fed_crate = imod->fedCh(ipair).fedCrate_;
	      uint16_t fed_slot  = imod->fedCh(ipair).fedSlot_;
	      uint16_t fed_id    = imod->fedCh(ipair).fedId_;

	      if ( fed_id ) { 

		num_of_devices.nFedChans_++;
		
		std::vector<uint16_t>::iterator icrate = find( fed_crates.begin(), fed_crates.end(), fed_crate );
		if ( icrate == fed_crates.end() ) { 
		  num_of_devices.nFedCrates_++; 
		  fed_crates.push_back(fed_crate); 
		}
		
		std::vector<uint16_t>::iterator islot = find( fed_slots.begin(), fed_slots.end(), fed_slot );
		if ( islot == fed_slots.end() ) { 
		  num_of_devices.nFedSlots_++; 
		  fed_slots.push_back(fed_slot); 
		}

		std::vector<uint16_t>::iterator ifed = find( fed_ids.begin(), fed_ids.end(), fed_id );
		if ( ifed == fed_ids.end() ) { 
		  num_of_devices.nFedIds_++; 
		  fed_ids.push_back(fed_id); 
		}

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
void SiStripFecCabling::print( std::stringstream& ss ) const {
  uint32_t valid = 0;
  uint32_t total = 0;
  ss << "[SiStripFecCabling::" << __func__ << "] Printing FEC cabling:" << std::endl;
  ss << "Printing cabling for " << crates().size() << " crates" << std::endl;
  for ( std::vector<SiStripFecCrate>::const_iterator icrate = crates().begin(); icrate != crates().end(); icrate++ ) {
    ss << "Printing cabling for " << icrate->fecs().size() << " FECs for crate " << icrate->fecCrate() << std::endl;
    for ( std::vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++ ) {
      ss << "Printing cabling for " << ifec->rings().size() << " rings for FEC " << ifec->fecSlot() << std::endl;
      for ( std::vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
	ss << "Printing cabling for " << iring->ccus().size() << " CCUs for ring " << iring->fecRing() << std::endl;
	for ( std::vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
	  ss << "Printing cabling for " << iccu->modules().size() << " modules for CCU " << iccu->ccuAddr() << std::endl;
	  for ( std::vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); imod != iccu->modules().end(); imod++ ) {

	    SiStripModule::FedCabling conns = imod->fedChannels();
	    SiStripModule::FedCabling::const_iterator ii = conns.begin();
	    SiStripModule::FedCabling::const_iterator jj = conns.end();
	    for ( ; ii != jj; ++ii ) {
	      if ( ii->second.fedId_ != sistrip::invalid_ &&
		   ii->second.fedCh_ != sistrip::invalid_ ) { valid++; }
	      total++;
	    }
	    ss << *imod << std::endl;
	  } 
	}
      }
    }
  }
  ss << "Number of connected:   " << valid << std::endl
     << "Number of connections: " << total << std::endl;
}

// -----------------------------------------------------------------------------
//
void SiStripFecCabling::terse( std::stringstream& ss ) const {
  ss << "[SiStripFecCabling::" << __func__ << "] Printing FEC cabling:" << std::endl;
  for ( std::vector<SiStripFecCrate>::const_iterator icrate = crates().begin(); icrate != crates().end(); icrate++ ) {
    for ( std::vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++ ) {
      for ( std::vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
	for ( std::vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
	  for ( std::vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); imod != iccu->modules().end(); imod++ ) {
	    imod->terse(ss); 
	    ss << std::endl;
	  } 
	}
      }
    }
  }
}

// -----------------------------------------------------------------------------
//
std::ostream& operator<< ( std::ostream& os, const SiStripFecCabling& cabling ) {
  std::stringstream ss;
  cabling.print(ss);
  os << ss.str();
  return os;
}
