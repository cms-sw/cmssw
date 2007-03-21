// Last commit: $Id: $

#include "DQM/SiStripCommissioningDbClients/interface/VpspScanHistosUsingDb.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include <iostream>

using namespace std;

// -----------------------------------------------------------------------------
/** */
VpspScanHistosUsingDb::VpspScanHistosUsingDb( MonitorUserInterface* mui,
					      const DbParams& params )
  : VpspScanHistograms( mui ),
    CommissioningHistosUsingDb( params )
{
  cout << endl // LogTrace(mlDqmClient_) 
       << "[VpspScanHistosUsingDb::" << __func__ << "]"
       << " Constructing object..." << endl;
}

// -----------------------------------------------------------------------------
/** */
VpspScanHistosUsingDb::~VpspScanHistosUsingDb() {
  cout << endl // LogTrace(mlDqmClient_) 
       << "[VpspScanHistosUsingDb::" << __func__ << "]"
       << " Destructing object..." << endl;
}

// -----------------------------------------------------------------------------
/** */
void VpspScanHistosUsingDb::uploadToConfigDb() {
  
  if ( !db_ ) {
    cerr << endl // edm::LogWarning(mlDqmClient_) 
	 << "[VpspScanHistosUsingDb::" << __func__ << "]"
	 << " NULL pointer to SiStripConfigDb interface! Aborting upload..." << endl;
    return;
  }

  // Update all APV device descriptions with new VPSP settings
  db_->resetDeviceDescriptions();
  SiStripConfigDb::DeviceDescriptions devices;
  db_->getDeviceDescriptions( devices, APV25 );
  update( devices );
  db_->uploadDeviceDescriptions(false);
  cout << endl // LogTrace(mlDqmClient_) 
       << "[VpspScanHistosUsingDb::" << __func__ << "]"
       << "Upload of LLD settings to DB finished!" << endl;

}

// -----------------------------------------------------------------------------
/** */
void VpspScanHistosUsingDb::update( SiStripConfigDb::DeviceDescriptions& devices ) {
  
  // Iterate through devices and update device descriptions
  SiStripConfigDb::DeviceDescriptions::iterator idevice;
  for ( idevice = devices.begin(); idevice != devices.end(); idevice++ ) {
    
    // Check device type
    if ( (*idevice)->getDeviceType() != APV25 ) {
      cerr << endl // edm::LogWarning(mlDqmClient_) 
	   << "[VpspScanHistosUsingDb::" << __func__ << "]"
	   << " Unexpected device type: " 
	   << (*idevice)->getDeviceType() << endl;
      continue;
    }
    
    // Retrieve description
    apvDescription* desc = dynamic_cast<apvDescription*>( *idevice );
    if ( !desc ) {
      cerr << endl // edm::LogWarning(mlDqmClient_) 
	   << "[VpspScanHistosUsingDb::" << __func__ << "]"
	   << " Unable to dynamic cast to apvDescription*" << endl;
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
      
      cout << endl // LogTrace(mlDqmClient_) 
	   << "[VpspScanHistosUsingDb::" << __func__ << "]"
	   << " Initial VPSP setting: " << desc->getVpsp() << endl;
      
      if ( iapv == 0 ) { desc->setVpsp( iter->second.vpsp0() ); }
      if ( iapv == 1 ) { desc->setVpsp( iter->second.vpsp1() ); }
      
      cout << endl // LogTrace(mlDqmClient_) 
	   << "[VpspScanHistosUsingDb::" << __func__ << "]"
	   << " Updated VPSP setting: " << desc->getVpsp() << endl;
      
    } else {
      cerr << endl // LogTrace(mlDqmClient_) 
	   << "[VpspScanHistosUsingDb::" << __func__ << "]"
	   << " Unable to find PLL settings for device with params FEC/slot/ring/CCU/LLDchan/APV: " 
	   << fec_path.fecCrate_ << "/"
	   << fec_path.fecSlot_ << "/"
	   << fec_path.fecRing_ << "/"
	   << fec_path.ccuAddr_ << "/"
	   << fec_path.ccuChan_ << "/"
	   << fec_path.channel() << "/" << iapv
	   << endl;
    }
      
  }

}








  

