// Last commit: $Id: $

#include "DQM/SiStripCommissioningDbClients/interface/OptoScanHistosUsingDb.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include <iostream>

using namespace std;

// -----------------------------------------------------------------------------
/** */
OptoScanHistosUsingDb::OptoScanHistosUsingDb( MonitorUserInterface* mui,
					      const DbParams& params )
  : OptoScanHistograms( mui ),
    CommissioningHistosUsingDb( params )
{
  cout << endl // LogTrace(mlDqmClient_) 
       << "[OptoScanHistosUsingDb::" << __func__ << "]"
       << " Constructing object..." << endl;
}

// -----------------------------------------------------------------------------
/** */
OptoScanHistosUsingDb::~OptoScanHistosUsingDb() {
  cout << endl // LogTrace(mlDqmClient_) 
       << "[OptoScanHistosUsingDb::" << __func__ << "]"
       << " Destructing object..." << endl;
}

// -----------------------------------------------------------------------------
/** */
void OptoScanHistosUsingDb::uploadToConfigDb() {
  
  if ( !db_ ) {
    cerr << endl // edm::LogWarning(mlDqmClient_) 
	 << "[OptoScanHistosUsingDb::" << __func__ << "]"
	 << " NULL pointer to SiStripConfigDb interface! Aborting upload..." << endl;
    return;
  }

  // Update LLD descriptions with new bias/gain settings
  db_->resetDeviceDescriptions();
  SiStripConfigDb::DeviceDescriptions devices;
  db_->getDeviceDescriptions( devices, LASERDRIVER );
  update( devices );
  db_->uploadDeviceDescriptions(false);
  cout << endl // LogTrace(mlDqmClient_) 
       << "[OptoScanHistosUsingDb::" << __func__ << "]"
       << "Upload of LLD settings to DB finished!" << endl;

}

// -----------------------------------------------------------------------------
/** */
void OptoScanHistosUsingDb::update( SiStripConfigDb::DeviceDescriptions& devices ) {
  
  // Iterate through devices and update device descriptions
  SiStripConfigDb::DeviceDescriptions::iterator idevice;
  for ( idevice = devices.begin(); idevice != devices.end(); idevice++ ) {
    
    // Check device type
    if ( (*idevice)->getDeviceType() != LASERDRIVER ) {
      cerr << endl // edm::LogWarning(mlDqmClient_) 
	   << "[OptoScanHistosUsingDb::" << __func__ << "]"
	   << " Unexpected device type: " 
	   << (*idevice)->getDeviceType() << endl;
      continue;
    }
    
    // Cast to retrieve appropriate description object
    laserdriverDescription* desc = dynamic_cast<laserdriverDescription*>( *idevice );
    if ( !desc ) {
      cerr << endl // edm::LogWarning(mlDqmClient_) 
	   << "[OptoScanHistosUsingDb::" << __func__ << "]"
	   << " Unable to dynamic cast to laserdriverDescription*" << endl;
      continue;
    }
    
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
					ichan ).key();
      fec_path = SiStripFecKey( fec_key );
      
      // Iterate through all channels and extract LLD settings 
      map<uint32_t,OptoScanAnalysis>::const_iterator iter = data_.find( fec_key );
      if ( iter != data_.end() ) {

	cout << endl // LogTrace(mlDqmClient_) 
	     << "[OptoScanHistosUsingDb::" << __func__ << "]"
	     << " Initial bias/gain settings for LLD channel " << ichan << ": " 
	     << static_cast<uint16_t>(desc->getGain(ichan)) << "/" 
	     << static_cast<uint16_t>(desc->getBias(ichan));

	uint16_t gain = iter->second.gain();
	desc->setGain( ichan, gain );
	desc->setBias( ichan, iter->second.bias()[gain] );
	
	cout << endl // LogTrace(mlDqmClient_) 
	     << "[OptoScanHistosUsingDb::" << __func__ << "]"
	     << " Updated bias/gain settings for LLD channel " << ichan << ": " 
	     << static_cast<uint16_t>(desc->getGain(ichan)) << "/" 
	     << static_cast<uint16_t>(desc->getBias(ichan));
      
      } else {
	cerr << endl // edm::LogWarning(mlDqmClient_) 
	     << "[OptoScanHistosUsingDb::" << __func__ << "]"
	     << " Unable to find FEC key with params FEC/slot/ring/CCU/LLDchan: " 
	     << fec_path.fecCrate_ << "/"
	     << fec_path.fecSlot_ << "/"
	     << fec_path.fecRing_ << "/"
	     << fec_path.ccuAddr_ << "/"
	     << fec_path.ccuChan_ << "/"
	     << fec_path.channel();
      }
      
    }

  }

}








  
