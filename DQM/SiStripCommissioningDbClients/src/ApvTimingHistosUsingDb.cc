// Last commit: $Id: ApvTimingHistosUsingDb.cc,v 1.19 2008/03/06 13:30:52 delaer Exp $

#include "DQM/SiStripCommissioningDbClients/interface/ApvTimingHistosUsingDb.h"
#include "CondFormats/SiStripObjects/interface/ApvTimingAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
ApvTimingHistosUsingDb::ApvTimingHistosUsingDb( DQMOldReceiver* mui,
						SiStripConfigDb* const db ) 
  : CommissioningHistograms( mui, sistrip::APV_TIMING ),
    CommissioningHistosUsingDb( db, mui, sistrip::APV_TIMING ),
    ApvTimingHistograms( mui ),
    uploadFecSettings_(true),
    uploadFedSettings_(true)
{
  LogTrace(mlDqmClient_) 
    << "[ApvTimingHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
ApvTimingHistosUsingDb::ApvTimingHistosUsingDb( DQMStore* bei,
						SiStripConfigDb* const db ) 
  : CommissioningHistosUsingDb( db, sistrip::APV_TIMING ),
    ApvTimingHistograms( bei ),
    uploadFecSettings_(true),
    uploadFedSettings_(true)
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
void ApvTimingHistosUsingDb::uploadConfigurations() {
  LogTrace(mlDqmClient_) 
    << "[ApvTimingHistosUsingDb::" << __func__ << "]";
  
  if ( !db() ) {
    edm::LogError(mlDqmClient_) 
      << "[ApvTimingHistosUsingDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb interface!"
      << " Aborting upload...";
    return;
  }
  
  if ( uploadFecSettings_ ) {

    // Retrieve and update PLL device descriptions
    SiStripConfigDb::DeviceDescriptionsRange devices = db()->getDeviceDescriptions( PLL ); 
    bool upload = update( devices );
    
    // Check if new PLL settings are valid 
    if ( !upload ) {
      edm::LogError(mlDqmClient_) 
	<< "[ApvTimingHistosUsingDb::" << __func__ << "]"
	<< " Found invalid PLL settings (coarse > 15)"
	<< " Aborting update to database...";
      return;
    }
    
    // Upload PLL device descriptions
    if ( doUploadConf() ) { 
      edm::LogVerbatim(mlDqmClient_) 
	<< "[ApvTimingHistosUsingDb::" << __func__ << "]"
	<< " Uploading PLL settings to DB...";
      db()->uploadDeviceDescriptions(); 
      edm::LogVerbatim(mlDqmClient_) 
	<< "[ApvTimingHistosUsingDb::" << __func__ << "]"
	<< " Upload of PLL settings to DB finished!";
    } else {
      edm::LogWarning(mlDqmClient_) 
	<< "[ApvTimingHistosUsingDb::" << __func__ << "]"
	<< " TEST only! No PLL settings will be uploaded to DB...";
    }
    
  } else {
    LogTrace(mlDqmClient_) 
      << "[ApvTimingHistosUsingDb::" << __func__ << "]"
      << " No upload of PLL settings to DB, as defined by .cfg file!";
  }
  
  if ( uploadFedSettings_ ) {
    
    // Update FED descriptions with new ticker thresholds
    SiStripConfigDb::FedDescriptionsRange feds = db()->getFedDescriptions(); 
    update( feds );
    
    // Update FED descriptions with new ticker thresholds
    if ( doUploadConf() ) { 
      edm::LogVerbatim(mlDqmClient_) 
	<< "[ApvTimingHistosUsingDb::" << __func__ << "]"
	<< " Uploading FED ticker thresholds to DB...";
      db()->uploadFedDescriptions(); 
      edm::LogVerbatim(mlDqmClient_) 
	<< "[ApvTimingHistosUsingDb::" << __func__ << "]"
	<< " Upload of FED ticker thresholds to DB finished!";
    } else {
      edm::LogWarning(mlDqmClient_) 
	<< "[ApvTimingHistosUsingDb::" << __func__ << "]"
	<< " TEST only! No FED ticker thresholds will be uploaded to DB...";
    }
    
  } else {
    LogTrace(mlDqmClient_) 
      << "[ApvTimingHistosUsingDb::" << __func__ << "]"
      << " No Upload of FED ticker thresholds to DB, as defined by .cfg file!";
  }

}

// -----------------------------------------------------------------------------
/** */
bool ApvTimingHistosUsingDb::update( SiStripConfigDb::DeviceDescriptionsRange devices ) {
  
  // Iterate through devices and update device descriptions
  uint16_t updated = 0;
  std::vector<SiStripFecKey> invalid;
  SiStripConfigDb::DeviceDescriptionsV::const_iterator idevice;
  for ( idevice = devices.begin(); idevice != devices.end(); idevice++ ) {
    
    // Check device type
    if ( (*idevice)->getDeviceType() != PLL ) { continue; }
    
    // Cast to retrieve appropriate description object
    pllDescription* desc = dynamic_cast<pllDescription*>( *idevice ); 
    if ( !desc ) { continue; }
    
    // Retrieve device addresses from device description
    const SiStripConfigDb::DeviceAddress& addr = db()->deviceAddress(*desc);
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
      Analyses::const_iterator iter = data().find( fec_key );
      if ( iter != data().end() ) { 

	if ( !iter->second->isValid() ) { continue; }

	ApvTimingAnalysis* anal = dynamic_cast<ApvTimingAnalysis*>( iter->second );
	if ( !anal ) { 
	  edm::LogError(mlDqmClient_)
	    << "[ApvTimingHistosUsingDb::" << __func__ << "]"
	    << " NULL pointer to analysis object!";
	  continue; 
	}
	
	// Calculate coarse and fine delays
	uint32_t delay = static_cast<uint32_t>( rint( anal->delay() * 24. / 25. ) ); 
	coarse = static_cast<uint16_t>( desc->getDelayCoarse() ) 
	  + ( static_cast<uint16_t>( desc->getDelayFine() ) + delay ) / 24;
	fine = ( static_cast<uint16_t>( desc->getDelayFine() ) + delay ) % 24;
	
	// Record PPLs maximum coarse setting
	if ( coarse > 15 ) { invalid.push_back(fec_path); }
	
      } else {
	edm::LogWarning(mlDqmClient_) 
	  << "[ApvTimingHistosUsingDb::" << __func__ << "]"
	  << " Unable to find FEC key with params Crate/FEC/slot/ring/CCU/LLD: " 
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
	 << " for Crate/FEC/slot/ring/CCU "
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
      //LogTrace(mlDqmClient_) << ss.str();

    } else {
      LogTrace(mlDqmClient_) 
	<< "[ApvTimingHistosUsingDb::" << __func__ << "]"
	<< " Unexpected PLL delay settings for Crate/FEC/slot/ring/CCU " 
	<< fec_path.fecCrate() << "/"
	<< fec_path.fecSlot() << "/"
	<< fec_path.fecRing() << "/"
	<< fec_path.ccuAddr() << "/"
	<< fec_path.ccuChan();
    }

  }

  // Check if invalid settings were found
  if ( !invalid.empty() ) {
    std::stringstream ss;
    ss << "[ApvTimingHistosUsingDb::" << __func__ << "]"
       << " Found PLL coarse setting of 15" 
       << " (not allowed!) for following channels"
       << " (Crate/FEC/slot/ring/CCU/LLD): ";
    std::vector<SiStripFecKey>::iterator ikey = invalid.begin();
    std::vector<SiStripFecKey>::iterator jkey = invalid.end();
    for ( ; ikey != jkey; ++ikey ) {
      ss << ikey->fecCrate() << "/"
	 << ikey->fecSlot() << "/"
	 << ikey->fecRing() << "/"
	 << ikey->ccuAddr() << "/"
	 << ikey->ccuChan() << ", ";
    }
    edm::LogWarning(mlDqmClient_) << ss.str();
    return false;
  }
  
  edm::LogVerbatim(mlDqmClient_) 
    << "[ApvTimingHistosUsingDb::" << __func__ << "]"
    << " Updated PLL settings for " 
    << updated << " modules";
  return true;
    
}

// -----------------------------------------------------------------------------
/** */
void ApvTimingHistosUsingDb::update( SiStripConfigDb::FedDescriptionsRange feds ) {
  
  // Retrieve FED ids from cabling
  std::vector<uint16_t> ids = cabling()->feds() ;
  
  // Iterate through feds and update fed descriptions
  uint16_t updated = 0;
  SiStripConfigDb::FedDescriptionsV::const_iterator ifed;
  for ( ifed = feds.begin(); ifed != feds.end(); ifed++ ) {
    
    // If FED id not found in list (from cabling), then continue
    if ( find( ids.begin(), ids.end(), (*ifed)->getFedId() ) == ids.end() ) { continue; } 
    
    for ( uint16_t ichan = 0; ichan < sistrip::FEDCH_PER_FED; ichan++ ) {

      // Build FED and FEC keys
      const FedChannelConnection& conn = cabling()->connection( (*ifed)->getFedId(), ichan );
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
      Analyses::const_iterator iter = data().find( fec_key.key() );
      if ( iter != data().end() ) { 
	
	ApvTimingAnalysis* anal = dynamic_cast<ApvTimingAnalysis*>( iter->second );
	if ( !anal ) { 
	  edm::LogError(mlDqmClient_)
	    << "[ApvTimingHistosUsingDb::" << __func__ << "]"
	    << " NULL pointer to analysis object!";
	  continue; 
	}
	
	Fed9U::Fed9UAddress addr( ichan );
	std::stringstream ss;
	ss << "[ApvTimingHistosUsingDb::" << __func__ << "]"
	   << " Updating the frame-finding threshold"
	   << " for Crate/FEC/slot/ring/CCU "
	   << fec_key.fecCrate() << "/"
	   << fec_key.fecSlot() << "/"
	   << fec_key.fecRing() << "/"
	   << fec_key.ccuAddr() << "/"
	   << fec_key.ccuChan() 
	   << " and FED id/ch "
	   << fed_key.fedId() << "/"
	   << fed_key.fedChannel()
	   << " in loop FED id/ch " 
	   << (*ifed)->getFedId() << "/" << ichan
	   << " from "
	   << static_cast<uint16_t>( (*ifed)->getFrameThreshold( addr ) );
	if ( anal->frameFindingThreshold() < sistrip::valid_ ) {
	  (*ifed)->setFrameThreshold( addr, anal->frameFindingThreshold() );
	  updated++;
	  ss << " to "
	     << static_cast<uint16_t>( (*ifed)->getFrameThreshold( addr ) )
	     << " tick base/peak/height: " 
	     << anal->base() << "/"
	     << anal->peak() << "/"
	     << anal->height();
	} else { ss << " to same value! (Invalid returned!)"; }
	ss << std::endl; 
	anal->print(ss);
	//LogTrace(mlDqmClient_) << ss.str();
	
      } else {
	edm::LogWarning(mlDqmClient_) 
	  << "[ApvTimingHistosUsingDb::" << __func__ << "]"
	  << " Unable to find ticker thresholds for FedKey/Id/Ch: 0x" 
	  << hex << setw(8) << setfill('0') << fed_key.key() << dec << "/"
	  << (*ifed)->getFedId() << "/"
	  << ichan
	  << " and device with Crate/FEC/slot/ring/CCU/LLD " 
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
    << " Updated ticker thresholds for " << updated 
    << " channels on " << ids.size() << " FEDs!";
  
}

// -----------------------------------------------------------------------------
/** */
void ApvTimingHistosUsingDb::create( SiStripConfigDb::AnalysisDescriptionsV& desc,
				     Analysis analysis ) {

#ifdef USING_NEW_DATABASE_MODEL
  
  ApvTimingAnalysis* anal = dynamic_cast<ApvTimingAnalysis*>( analysis->second );
  if ( !anal ) { return; }
  
  SiStripFecKey fec_key( anal->fecKey() );
  SiStripFedKey fed_key( anal->fedKey() );
  
  for ( uint16_t iapv = 0; iapv < 2; ++iapv ) {
    
    // Create description
    TimingAnalysisDescription* tmp;
    tmp = new TimingAnalysisDescription( anal->time(),
					 anal->refTime(),
					 anal->delay(),
					 anal->height(),
					 anal->base(),
					 anal->peak(),
					 anal->frameFindingThreshold(),
					 anal->optimumSamplingPoint(),
					 ApvTimingAnalysis::tickMarkHeightThreshold_,
					 true, //@@ APV timing analysis (not FED timing)
					 fec_key.fecCrate(),
					 fec_key.fecSlot(),
					 fec_key.fecRing(),
					 fec_key.ccuAddr(),
					 fec_key.ccuChan(),
					 SiStripFecKey::i2cAddr( fec_key.lldChan(), !iapv ), 
					 db()->dbParams().partitions().begin()->second.partitionName(),
					 db()->dbParams().partitions().begin()->second.runNumber(),
					 anal->isValid(),
					 "",
					 fed_key.fedId(),
					 fed_key.feUnit(),
					 fed_key.feChan(),
					 fed_key.fedApv() );
    
    // Add comments
    typedef std::vector<std::string> Strings;
    Strings errors = anal->getErrorCodes();
    Strings::const_iterator istr = errors.begin();
    Strings::const_iterator jstr = errors.end();
    for ( ; istr != jstr; ++istr ) { tmp->addComments( *istr ); }
    
    // Store description
    desc.push_back( tmp );
    
  }

#endif
  
}
