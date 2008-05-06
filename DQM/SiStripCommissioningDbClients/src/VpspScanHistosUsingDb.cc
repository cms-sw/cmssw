// Last commit: $Id: VpspScanHistosUsingDb.cc,v 1.15 2008/03/06 13:30:53 delaer Exp $

#include "DQM/SiStripCommissioningDbClients/interface/VpspScanHistosUsingDb.h"
#include "CondFormats/SiStripObjects/interface/VpspScanAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
VpspScanHistosUsingDb::VpspScanHistosUsingDb( DQMOldReceiver* mui,
					      SiStripConfigDb* const db ) 
  : CommissioningHistograms( mui, sistrip::VPSP_SCAN ),
    CommissioningHistosUsingDb( db, sistrip::VPSP_SCAN ),
    VpspScanHistograms( mui )
{
  LogTrace(mlDqmClient_) 
    << "[VpspScanHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
VpspScanHistosUsingDb::VpspScanHistosUsingDb( DQMStore* bei,
					      SiStripConfigDb* const db ) 
  : CommissioningHistosUsingDb( db, sistrip::VPSP_SCAN ),
    VpspScanHistograms( bei )
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
void VpspScanHistosUsingDb::uploadConfigurations() {
  LogTrace(mlDqmClient_) 
    << "[VpspScanHistosUsingDb::" << __func__ << "]";
  
  if ( !db() ) {
    edm::LogError(mlDqmClient_) 
      << "[VpspScanHistosUsingDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb interface!"
      << " Aborting upload...";
    return;
  }
  
  // Update all APV device descriptions with new VPSP settings
  SiStripConfigDb::DeviceDescriptionsRange devices = db()->getDeviceDescriptions();
  update( devices );
  if ( doUploadConf() ) { 
    edm::LogVerbatim(mlDqmClient_) 
      << "[VpspScanHistosUsingDb::" << __func__ << "]"
      << " Uploading VPSP settings to DB...";
    db()->uploadDeviceDescriptions(); 
    edm::LogVerbatim(mlDqmClient_) 
      << "[VpspScanHistosUsingDb::" << __func__ << "]"
      << " Uploaded VPSP settings to DB!";
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
void VpspScanHistosUsingDb::update( SiStripConfigDb::DeviceDescriptionsRange devices ) {
  
  // Iterate through devices and update device descriptions
  SiStripConfigDb::DeviceDescriptionsV::const_iterator idevice;
  for ( idevice = devices.begin(); idevice != devices.end(); idevice++ ) {
    
    // Check device type
    if ( (*idevice)->getDeviceType() != APV25 ) { continue; }
    
    // Cast to retrieve appropriate description object
    apvDescription* desc = dynamic_cast<apvDescription*>( *idevice );
    if ( !desc ) { continue; }
    
    // Retrieve device addresses from device description
    const SiStripConfigDb::DeviceAddress& addr = db()->deviceAddress(*desc);
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
    Analyses::const_iterator iter = data().find( fec_key );
    if ( iter != data().end() ) {

      if ( !iter->second->isValid() ) { continue; }

      VpspScanAnalysis* anal = dynamic_cast<VpspScanAnalysis*>( iter->second );
      if ( !anal ) { 
	edm::LogError(mlDqmClient_)
	  << "[VpspScanHistosUsingDb::" << __func__ << "]"
	  << " NULL pointer to analysis object!";
	continue; 
      }
      
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
      if ( iapv == 0 ) { desc->setVpsp( anal->vpsp()[0] ); }
      if ( iapv == 1 ) { desc->setVpsp( anal->vpsp()[1] ); }
      ss << " to " << static_cast<uint16_t>(desc->getVpsp());
      LogTrace(mlDqmClient_) << ss.str();
      
    } else {
      edm::LogWarning(mlDqmClient_) 
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

// -----------------------------------------------------------------------------
/** */
void VpspScanHistosUsingDb::create( SiStripConfigDb::AnalysisDescriptionsV& desc,
				    Analysis analysis ) {

#ifdef USING_NEW_DATABASE_MODEL
  
  VpspScanAnalysis* anal = dynamic_cast<VpspScanAnalysis*>( analysis->second );
  if ( !anal ) { return; }
  
  SiStripFecKey fec_key( anal->fecKey() ); 
  SiStripFedKey fed_key( anal->fedKey() );
  
  for ( uint16_t iapv = 0; iapv < 2; ++iapv ) {

    // Create description
    VpspScanAnalysisDescription* tmp;
    tmp = new VpspScanAnalysisDescription( anal->vpsp()[iapv],
					   anal->adcLevel()[iapv],
					   anal->fraction()[iapv],
					   anal->topEdge()[iapv],
					   anal->bottomEdge()[iapv],
					   anal->topLevel()[iapv],
					   anal->bottomLevel()[iapv],
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

