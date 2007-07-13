// Last commit: $Id: VpspScanHistosUsingDb.cc,v 1.6 2007/06/19 12:30:37 bainbrid Exp $

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
    db_->uploadDeviceDescriptions(true); 
  } else {
    edm::LogWarning(mlDqmClient_) 
      << "[VpspScanHistosUsingDb::" << __func__ << "]"
      << " TEST only! No VPSP settings will be uploaded to DB...";
  }
  LogTrace(mlDqmClient_) 
    << "[VpspScanHistosUsingDb::" << __func__ << "]"
    << " Upload of VPSP settings to DB finished!";
  
}

// -----------------------------------------------------------------------------
/** */
void VpspScanHistosUsingDb::update( SiStripConfigDb::DeviceDescriptions& devices ) {
  
  // Iterate through devices and update device descriptions
  SiStripConfigDb::DeviceDescriptions::iterator idevice;
  for ( idevice = devices.begin(); idevice != devices.end(); idevice++ ) {
    
    // Check device type
    if ( (*idevice)->getDeviceType() != APV25 ) { continue; }
    
    // Cast to retrieve appropriate description object
    apvDescription* desc = dynamic_cast<apvDescription*>( *idevice );
    if ( !desc ) { continue; }
    
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
				      ichan+1 ).key();
    fec_path = SiStripFecKey( fec_key );
      
    // Iterate through all channels and extract LLD settings 
    map<uint32_t,VpspScanAnalysis*>::const_iterator iter = data_.find( fec_key );
    if ( iter != data_.end() ) {
      if ( iter->second->isValid() ) {
	std::stringstream ss;
	ss << "[VpspScanHistosUsingDb::" << __func__ << "]"
	   << " Updating VPSP setting for crate/FEC/slot/ring/CCU/LLD/APV " 
	   << fec_path.fecCrate() << "/"
	   << fec_path.fecSlot() << "/"
	   << fec_path.fecRing() << "/"
	   << fec_path.ccuAddr() << "/"
	   << fec_path.ccuChan() << "/"
	   << fec_path.channel() 
	   << iapv
	   << " from "
	   << static_cast<uint16_t>(desc->getVpsp());
	if ( iapv == 0 ) { desc->setVpsp( iter->second->vpsp()[0] ); }
	if ( iapv == 1 ) { desc->setVpsp( iter->second->vpsp()[1] ); }
	ss << " to "
	   << static_cast<uint16_t>(desc->getVpsp());
	LogTrace(mlDqmClient_) << ss.str();
      } else {
	std::stringstream ss;
	ss << "[VpspScanHistosUsingDb::" << __func__ << "]"
	   << " Invalid analysis!" << std::endl; 
	iter->second->print( ss, 1 );
	iter->second->print( ss, 2 );
	edm::LogWarning(mlDqmClient_) << ss.str(); 
      }
      
    } else {
      LogTrace(mlDqmClient_) 
	<< "[VpspScanHistosUsingDb::" << __func__ << "]"
	<< " Unable to find FEC key with params FEC/slot/ring/CCU/LLDchan/APV: " 
	<< fec_path.fecCrate() << "/"
	<< fec_path.fecSlot() << "/"
	<< fec_path.fecRing() << "/"
	<< fec_path.ccuAddr() << "/"
	<< fec_path.ccuChan() << "/"
	<< fec_path.channel() << "/" 
	<< iapv+1;

    }
      
  }

}








  

