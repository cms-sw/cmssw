// Last commit: $Id: ApvTimingHistosUsingDb.cc,v 1.4 2007/05/24 15:59:49 bainbrid Exp $

#include "DQM/SiStripCommissioningDbClients/interface/ApvTimingHistosUsingDb.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
ApvTimingHistosUsingDb::ApvTimingHistosUsingDb( MonitorUserInterface* mui,
						const DbParams& params )
  : ApvTimingHistograms( mui ),
    CommissioningHistosUsingDb( params )
{
  LogTrace(mlDqmClient_) 
    << "[ApvTimingHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
ApvTimingHistosUsingDb::ApvTimingHistosUsingDb( MonitorUserInterface* mui,
						SiStripConfigDb* const db ) 
  : ApvTimingHistograms( mui ),
    CommissioningHistosUsingDb( db )
{
  LogTrace(mlDqmClient_) 
    << "[ApvTimingHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
ApvTimingHistosUsingDb::ApvTimingHistosUsingDb( DaqMonitorBEInterface* bei,
						SiStripConfigDb* const db ) 
  : ApvTimingHistograms( bei ),
    CommissioningHistosUsingDb( db )
{
  LogTrace(mlDqmClient_) 
    << "[ApvTimingHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
ApvTimingHistosUsingDb::~ApvTimingHistosUsingDb() {
  LogTrace(mlDqmClient_) 
    << "[ApvTimingHistosUsingDb::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
void ApvTimingHistosUsingDb::uploadToConfigDb() {
  
  if ( !db_ ) {
    edm::LogWarning(mlDqmClient_) 
      << "[ApvTimingHistosUsingDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb interface!"
      << " Aborting upload...";
    return;
  }
  
  // Update PLL device descriptions
  db_->resetDeviceDescriptions();
  const SiStripConfigDb::DeviceDescriptions& devices = db_->getDeviceDescriptions(); 
  update( const_cast<SiStripConfigDb::DeviceDescriptions&>(devices) );
  if ( !test_ ) { 
    LogTrace(mlDqmClient_) 
      << "[ApvTimingHistosUsingDb::" << __func__ << "]"
      << " Uploading PLL settings to DB...";
    db_->uploadDeviceDescriptions(true); 
    LogTrace(mlDqmClient_) 
      << "[ApvTimingHistosUsingDb::" << __func__ << "]"
      << "Upload of PLL settings to DB finished!";
  } else {
    edm::LogWarning(mlDqmClient_) 
      << "[ApvTimingHistosUsingDb::" << __func__ << "]"
      << " TEST only! No PLL settings will be uploaded to DB...";
  }

  // Update FED descriptions with new ticker thresholds
  db_->resetFedDescriptions();
  const SiStripConfigDb::FedDescriptions& feds = db_->getFedDescriptions(); 
  update( const_cast<SiStripConfigDb::FedDescriptions&>(feds) );
  if ( !test_ ) { 
    LogTrace(mlDqmClient_) 
      << "[ApvTimingHistosUsingDb::" << __func__ << "]"
      << " Uploading FED ticker thresholds to DB...";
    db_->uploadFedDescriptions(false); 
    LogTrace(mlDqmClient_) 
      << "[ApvTimingHistosUsingDb::" << __func__ << "]"
      << " Upload of FED ticker thresholds to DB finished!";
  } else {
    edm::LogWarning(mlDqmClient_) 
      << "[ApvTimingHistosUsingDb::" << __func__ << "]"
      << " TEST only! No FED ticker thresholds will be uploaded to DB...";
  }
  
}

// -----------------------------------------------------------------------------
/** */
void ApvTimingHistosUsingDb::update( SiStripConfigDb::DeviceDescriptions& devices ) {

  // Iterate through devices and update device descriptions
  uint16_t updated = 0;
  SiStripConfigDb::DeviceDescriptions::iterator idevice;
  for ( idevice = devices.begin(); idevice != devices.end(); idevice++ ) {
    
    // Check device type
    if ( (*idevice)->getDeviceType() != PLL ) { continue; }
    
    // Cast to retrieve appropriate description object
    pllDescription* desc = dynamic_cast<pllDescription*>( *idevice ); 
    if ( !desc ) { continue; }
    
    // Retrieve device addresses from device description
    const SiStripConfigDb::DeviceAddress& addr = db_->deviceAddress(*desc);
    SiStripFecKey fec_path;
    
    // PLL delay settings
    uint32_t coarse = sistrip::invalid_; 
    uint32_t fine = sistrip::invalid_; 
    
    // Iterate through LLD channels
    for ( uint16_t ichan = 0; ichan < sistrip::CHANS_PER_LLD; ichan++ ) {
      
      // Construct key from device description
      uint32_t fec_key = SiStripFecKey( addr.fecCrate_,
					addr.fecSlot_, 
					addr.fecRing_,
					addr.ccuAddr_, 
					addr.ccuChan_,
					ichan+1 ).key();
      fec_path = SiStripFecKey( fec_key );
      
      // Locate appropriate analysis object    
      map<uint32_t,ApvTimingAnalysis*>::const_iterator iter = data_.find( fec_key );
      if ( iter != data_.end() ) { 
	
	// Check delay value
	if ( iter->second->refTime() < 0. || iter->second->refTime() > sistrip::maximum_ ) { 
	  edm::LogWarning(mlDqmClient_) 
	    << "[ApvTimingHistosUsingDb::" << __func__ << "]"
	    << " Unexpected maximum time setting: "
	    << iter->second->refTime();
	  continue;
	}
	
	// Check delay and tick height are valid
	if ( iter->second->delay() < 0. || 
	     iter->second->delay() > sistrip::maximum_ ) { 
	  edm::LogWarning(mlDqmClient_) 
	    << "[ApvTimingHistosUsingDb::" << __func__ << "]"
	    << " Unexpected delay value: "
	    << iter->second->delay();
	  continue; 
	}
	if ( iter->second->height() < 100. ) { 
	  edm::LogWarning(mlDqmClient_) 
	    << "[ApvTimingHistosUsingDb::" << __func__ << "]"
	    << " Unexpected tick height: "
	    << iter->second->height();
	  continue; 
	}
	
	// Calculate coarse and fine delays
	uint32_t delay = static_cast<uint32_t>( rint( iter->second->delay() * 24. / 25. ) ); 
	coarse = static_cast<uint16_t>( desc->getDelayCoarse() ) 
	  + ( static_cast<uint16_t>( desc->getDelayFine() ) + delay ) / 24;
	fine = ( static_cast<uint16_t>( desc->getDelayFine() ) + delay ) % 24;
	
      } else {
	edm::LogWarning(mlDqmClient_) 
	  << "[ApvTimingHistosUsingDb::" << __func__ << "]"
	  << " Unable to find FEC key with params FEC/slot/ring/CCU/LLD: " 
	  << fec_path.fecCrate() << "/"
	  << fec_path.fecSlot() << "/"
	  << fec_path.fecRing() << "/"
	  << fec_path.ccuAddr() << "/"
	  << fec_path.ccuChan() << "/"
	  << fec_path.channel();
      }

      // Exit LLD channel loop if coarse and fine delays are known
      if ( coarse != sistrip::invalid_ && 
	   fine != sistrip::invalid_ ) { break; }
      
    } // lld channel loop
    
	// Update PLL settings
    if ( coarse != sistrip::invalid_ && 
	 fine != sistrip::invalid_ ) { 
      
      std::stringstream ss;
      ss << "[ApvTimingHistosUsingDb::" << __func__ << "]"
	 << " Updating coarse/fine PLL settings"
	 << " for crate/FEC/slot/ring/CCU "
	 << fec_path.fecCrate() << "/"
	 << fec_path.fecSlot() << "/"
	 << fec_path.fecRing() << "/"
	 << fec_path.ccuAddr() << "/"
	 << fec_path.ccuChan() 
	 << " from "
	 << static_cast<uint16_t>( desc->getDelayCoarse() ) << "/" 
	 << static_cast<uint16_t>( desc->getDelayFine() );
      desc->setDelayCoarse(coarse);
      desc->setDelayFine(fine);
      updated++;
      ss << " to "
	 << static_cast<uint16_t>( desc->getDelayCoarse() ) << "/" 
	 << static_cast<uint16_t>( desc->getDelayFine() );
      LogTrace(mlDqmClient_) << ss.str();

    } else {
      LogTrace(mlDqmClient_) 
	<< "[ApvTimingHistosUsingDb::" << __func__ << "]"
	<< " Unexpected PLL delay settings for crate/FEC/slot/ring/CCU " 
	<< fec_path.fecCrate() << "/"
	<< fec_path.fecSlot() << "/"
	<< fec_path.fecRing() << "/"
	<< fec_path.ccuAddr() << "/"
	<< fec_path.ccuChan();
    }

  }
  
  edm::LogVerbatim(mlDqmClient_) 
    << "[ApvTimingHistosUsingDb::" << __func__ << "]"
    << " Updated PLL settings for " 
    << updated << " modules";
    
}

// -----------------------------------------------------------------------------
/** */
void ApvTimingHistosUsingDb::update( SiStripConfigDb::FedDescriptions& feds ) {
  
  // Iterate through feds and update fed descriptions
  uint16_t updated = 0;
  SiStripConfigDb::FedDescriptions::iterator ifed;
  for ( ifed = feds.begin(); ifed != feds.end(); ifed++ ) {
    
    for ( uint16_t ichan = 0; ichan < sistrip::FEDCH_PER_FED; ichan++ ) {

      // Build FED and FEC keys
      const FedChannelConnection& conn = cabling_->connection( (*ifed)->getFedId(), ichan );
      if ( conn.fecCrate() == sistrip::invalid_ ||
	   conn.fecSlot() == sistrip::invalid_ ||
	   conn.fecRing() == sistrip::invalid_ ||
	   conn.ccuAddr() == sistrip::invalid_ ||
	   conn.ccuChan() == sistrip::invalid_ ||
	   conn.lldChannel() == sistrip::invalid_ ) { continue; }
      SiStripFedKey fed_key( conn.fedId(), 
			     SiStripFedKey::feUnit( conn.fedCh() ),
			     SiStripFedKey::feChan( conn.fedCh() ) );
      SiStripFecKey fec_key( conn.fecCrate(), 
			     conn.fecSlot(), 
			     conn.fecRing(), 
			     conn.ccuAddr(), 
			     conn.ccuChan(), 
			     conn.lldChannel() );
      
      // Locate appropriate analysis object 
      map<uint32_t,ApvTimingAnalysis*>::const_iterator iter = data_.find( fec_key.key() );
      if ( iter != data_.end() ) { 
	uint32_t thresh = static_cast<uint32_t>( iter->second->base() + 
						 iter->second->height()*(2./3.) );
	Fed9U::Fed9UAddress addr( ichan );
	(*ifed)->setFrameThreshold( addr, thresh );
	updated++;
      } else {
	edm::LogWarning(mlDqmClient_) 
	  << "[ApvTimingHistosUsingDb::" << __func__ << "]"
	  << " Unable to find ticker thresholds for FedKey/Id/Ch: " 
	  << hex << setw(8) << setfill('0') << fed_key.key() << dec << "/"
	  << (*ifed)->getFedId() << "/"
	  << ichan
	  << " and device with FEC/slot/ring/CCU/LLD " 
	  << fec_key.fecCrate() << "/"
	  << fec_key.fecSlot() << "/"
	  << fec_key.fecRing() << "/"
	  << fec_key.ccuAddr() << "/"
	  << fec_key.ccuChan() << "/"
	  << fec_key.channel();
      }

    }
  }

  edm::LogVerbatim(mlDqmClient_) 
    << "[ApvTimingHistosUsingDb::" << __func__ << "]"
    << " Updated FED ticker thresholds for " 
    << updated << " channels";
  
}
