// Last commit: $Id: OptoScanHistosUsingDb.cc,v 1.5 2007/06/12 08:23:35 bainbrid Exp $

#include "DQM/SiStripCommissioningDbClients/interface/OptoScanHistosUsingDb.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
OptoScanHistosUsingDb::OptoScanHistosUsingDb( MonitorUserInterface* mui,
					      const DbParams& params )
  : OptoScanHistograms( mui ),
    CommissioningHistosUsingDb( params )
{
  LogTrace(mlDqmClient_) 
    << "[OptoScanHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
OptoScanHistosUsingDb::OptoScanHistosUsingDb( MonitorUserInterface* mui,
					      SiStripConfigDb* const db )
  : OptoScanHistograms( mui ),
    CommissioningHistosUsingDb( db )
{
  LogTrace(mlDqmClient_) 
    << "[OptoScanHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
OptoScanHistosUsingDb::OptoScanHistosUsingDb( DaqMonitorBEInterface* bei,
					      SiStripConfigDb* const db ) 
  : OptoScanHistograms( bei ),
    CommissioningHistosUsingDb( db )
{
  LogTrace(mlDqmClient_) 
    << "[OptoScanHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
OptoScanHistosUsingDb::~OptoScanHistosUsingDb() {
  LogTrace(mlDqmClient_) 
    << "[OptoScanHistosUsingDb::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
void OptoScanHistosUsingDb::uploadToConfigDb() {
  
  if ( !db_ ) {
    edm::LogWarning(mlDqmClient_) 
      << "[OptoScanHistosUsingDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb interface!"
      << " Aborting upload...";
    return;
  }

  // Update LLD descriptions with new bias/gain settings
  db_->resetDeviceDescriptions();
  const SiStripConfigDb::DeviceDescriptions& devices = db_->getDeviceDescriptions(); 
  update( const_cast<SiStripConfigDb::DeviceDescriptions&>(devices) );
  if ( !test_ ) { 
    LogTrace(mlDqmClient_) 
      << "[OptoScanHistosUsingDb::" << __func__ << "]"
      << " Uploading LLD settings to DB...";
    db_->uploadDeviceDescriptions(true); 
    LogTrace(mlDqmClient_) 
      << "[OptoScanHistosUsingDb::" << __func__ << "]"
      << " Upload of LLD settings to DB finished!";
  } else {
    edm::LogWarning(mlDqmClient_) 
      << "[OptoScanHistosUsingDb::" << __func__ << "]"
      << " TEST only! No LLD settings will be uploaded to DB...";
  }
  
}

// -----------------------------------------------------------------------------
/** */
void OptoScanHistosUsingDb::update( SiStripConfigDb::DeviceDescriptions& devices ) {
  
  // Iterate through devices and update device descriptions
  uint16_t updated = 0;
  SiStripConfigDb::DeviceDescriptions::iterator idevice;
  for ( idevice = devices.begin(); idevice != devices.end(); idevice++ ) {
    
    // Check device type
    if ( (*idevice)->getDeviceType() != LASERDRIVER ) { continue; }
    
    // Cast to retrieve appropriate description object
    laserdriverDescription* desc = dynamic_cast<laserdriverDescription*>( *idevice );
    if ( !desc ) { continue; }
    
    // Retrieve device addresses from device description
    const SiStripConfigDb::DeviceAddress& addr = db_->deviceAddress(*desc);
    SiStripFecKey fec_path;
    
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
      
      // Iterate through all channels and extract LLD settings 
      map<uint32_t,OptoScanAnalysis*>::const_iterator iter = data_.find( fec_key );
      if ( iter != data_.end() ) {

	// Check if analysis is valid
	if ( !iter->second->isValid() ) { continue; }

	uint16_t gain = iter->second->gain();
	std::stringstream ss;
	ss << "[OptoScanHistosUsingDb::" << __func__ << "]"
	   << " Updating gain/bias LLD settings for crate/FEC/slot/ring/CCU/LLD "
	   << fec_path.fecCrate() << "/"
	   << fec_path.fecSlot() << "/"
	   << fec_path.fecRing() << "/"
	   << fec_path.ccuAddr() << "/"
	   << fec_path.ccuChan() << "/"
	   << fec_path.channel() 
	   << " from "
	   << static_cast<uint16_t>( desc->getGain(ichan) ) << "/" 
	   << static_cast<uint16_t>( desc->getBias(ichan) );
	desc->setGain( ichan, gain );
	desc->setBias( ichan, iter->second->bias()[gain] );
	updated++;
	ss << " to "
	   << static_cast<uint16_t>(desc->getGain(ichan)) << "/" 
	   << static_cast<uint16_t>(desc->getBias(ichan));
	LogTrace(mlDqmClient_) << ss.str();
	
      } else {
	edm::LogWarning(mlDqmClient_) 
	  << "[OptoScanHistosUsingDb::" << __func__ << "]"
	  << " Unable to find FEC key with params FEC/slot/ring/CCU/LLD " 
	  << fec_path.fecCrate() << "/"
	  << fec_path.fecSlot() << "/"
	  << fec_path.fecRing() << "/"
	  << fec_path.ccuAddr() << "/"
	  << fec_path.ccuChan() << "/"
	  << fec_path.channel();
      }
      
    }

  }

  edm::LogVerbatim(mlDqmClient_) 
    << "[OptoScanHistosUsingDb::" << __func__ << "]"
    << " Updated PLL settings for " 
    << updated << " modules";
  

}








  
