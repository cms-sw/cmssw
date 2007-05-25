// Last commit: $Id: VpspScanHistosUsingDb.cc,v 1.4 2007/05/24 15:59:49 bainbrid Exp $

#include "DQM/SiStripCommissioningDbClients/interface/VpspScanHistosUsingDb.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
VpspScanHistosUsingDb::VpspScanHistosUsingDb( MonitorUserInterface* mui,
					      const DbParams& params )
  : VpspScanHistograms( mui ),
    CommissioningHistosUsingDb( params )
{
  LogTrace(mlDqmClient_) 
    << "[VpspScanHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
VpspScanHistosUsingDb::VpspScanHistosUsingDb( MonitorUserInterface* mui,
					      SiStripConfigDb* const db ) 
  : VpspScanHistograms( mui ),
    CommissioningHistosUsingDb( db )
{
  LogTrace(mlDqmClient_) 
    << "[VpspScanHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
VpspScanHistosUsingDb::VpspScanHistosUsingDb( DaqMonitorBEInterface* bei,
					      SiStripConfigDb* const db ) 
  : VpspScanHistograms( bei ),
    CommissioningHistosUsingDb( db )
{
  LogTrace(mlDqmClient_) 
    << "[VpspScanHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
VpspScanHistosUsingDb::~VpspScanHistosUsingDb() {
  LogTrace(mlDqmClient_) 
    << "[VpspScanHistosUsingDb::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
void VpspScanHistosUsingDb::uploadToConfigDb() {
  
  if ( !db_ ) {
    edm::LogWarning(mlDqmClient_) 
      << "[VpspScanHistosUsingDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb interface!"
      << " Aborting upload...";
    return;
  }
  
  // Update all APV device descriptions with new VPSP settings
  db_->resetDeviceDescriptions();
  const SiStripConfigDb::DeviceDescriptions& devices = db_->getDeviceDescriptions();
  update( const_cast<SiStripConfigDb::DeviceDescriptions&>(devices) );
  if ( !test_ ) { 
    LogTrace(mlDqmClient_) 
      << "[VpspScanHistosUsingDb::" << __func__ << "]"
      << " Uploading VPSP settings to DB...";
    db_->uploadDeviceDescriptions(false); 
  } else {
    edm::LogWarning(mlDqmClient_) 
      << "[PedestalsHistosUsingDb::" << __func__ << "]"
      << " TEST only! No VPSP settings will be uploaded to DB...";
  }
  LogTrace(mlDqmClient_) 
    << "[VpspScanHistosUsingDb::" << __func__ << "]"
    << "Upload of VPSP settings to DB finished!";
  
}

// -----------------------------------------------------------------------------
/** */
void VpspScanHistosUsingDb::update( SiStripConfigDb::DeviceDescriptions& devices ) {
  
  // Iterate through devices and update device descriptions
  SiStripConfigDb::DeviceDescriptions::iterator idevice;
  for ( idevice = devices.begin(); idevice != devices.end(); idevice++ ) {
    
    // Check device type
    if ( (*idevice)->getDeviceType() != APV25 ) {
      edm::LogWarning(mlDqmClient_) 
	<< "[VpspScanHistosUsingDb::" << __func__ << "]"
	<< " Unexpected device type: " 
	<< (*idevice)->getDeviceType();
      continue;
    }
    
    // Retrieve description
    apvDescription* desc = dynamic_cast<apvDescription*>( *idevice );
    if ( !desc ) {
      edm::LogWarning(mlDqmClient_) 
	<< "[VpspScanHistosUsingDb::" << __func__ << "]"
	<< " Unable to dynamic cast to apvDescription*";
      continue;
    }
    
    // Retrieve device addresses from device description
    const SiStripConfigDb::DeviceAddress& addr = db_->deviceAddress(*desc);
    SiStripFecKey fec_path;
    
    // Retrieve LLD channel and APV numbers
    uint16_t ichan = ( desc->getAddress() - 0x20 ) / 2;
    uint16_t iapv  = ( desc->getAddress() - 0x20 ) % 2;
    
    // Construct key from device description
    uint32_t fec_key = SiStripFecKey( addr.fecCrate_, 
				      addr.fecSlot_, 
				      addr.fecRing_, 
				      addr.ccuAddr_, 
				      addr.ccuChan_,
				      ichan ).key();
    fec_path = SiStripFecKey( fec_key );
      
    // Iterate through all channels and extract LLD settings 
    map<uint32_t,VpspScanAnalysis>::const_iterator iter = data_.find( fec_key );
    if ( iter != data_.end() ) {
      
      LogTrace(mlDqmClient_) 
	<< "[VpspScanHistosUsingDb::" << __func__ << "]"
	<< " Initial VPSP setting: " << desc->getVpsp();
      
      if ( iapv == 0 ) { desc->setVpsp( iter->second.vpsp()[0] ); }
      if ( iapv == 1 ) { desc->setVpsp( iter->second.vpsp()[1] ); }
      
      LogTrace(mlDqmClient_) 
	<< "[VpspScanHistosUsingDb::" << __func__ << "]"
	<< " Updated VPSP setting: " << desc->getVpsp();
      
    } else {
      LogTrace(mlDqmClient_) 
	<< "[VpspScanHistosUsingDb::" << __func__ << "]"
	<< " Unable to find PLL settings for device with params FEC/slot/ring/CCU/LLDchan/APV: " 
	<< fec_path.fecCrate() << "/"
	<< fec_path.fecSlot() << "/"
	<< fec_path.fecRing() << "/"
	<< fec_path.ccuAddr() << "/"
	<< fec_path.ccuChan() << "/"
	<< fec_path.channel() << "/" << iapv;

    }
      
  }

}








  

