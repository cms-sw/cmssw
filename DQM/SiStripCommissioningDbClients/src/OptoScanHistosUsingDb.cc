// Last commit: $Id: OptoScanHistosUsingDb.cc,v 1.8 2007/12/19 18:18:11 bainbrid Exp $

#include "DQM/SiStripCommissioningDbClients/interface/OptoScanHistosUsingDb.h"
#include "CondFormats/SiStripObjects/interface/OptoScanAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
OptoScanHistosUsingDb::OptoScanHistosUsingDb( MonitorUserInterface* mui,
					      const DbParams& params )
  : CommissioningHistosUsingDb( params ),
    OptoScanHistograms( mui )
{
  LogTrace(mlDqmClient_) 
    << "[OptoScanHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
OptoScanHistosUsingDb::OptoScanHistosUsingDb( MonitorUserInterface* mui,
					      SiStripConfigDb* const db )
  : CommissioningHistograms( mui, sistrip::OPTO_SCAN ),
    CommissioningHistosUsingDb( db, mui, sistrip::OPTO_SCAN ),
    OptoScanHistograms( mui )
{
  LogTrace(mlDqmClient_) 
    << "[OptoScanHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
OptoScanHistosUsingDb::OptoScanHistosUsingDb( DaqMonitorBEInterface* bei,
					      SiStripConfigDb* const db ) 
  : CommissioningHistosUsingDb( db, sistrip::OPTO_SCAN ),
    OptoScanHistograms( bei )
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
void OptoScanHistosUsingDb::uploadConfigurations() {
  
  if ( !db() ) {
    edm::LogError(mlDqmClient_) 
      << "[OptoScanHistosUsingDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb interface!"
      << " Aborting upload...";
    return;
  }

  // Update LLD descriptions with new bias/gain settings
  const SiStripConfigDb::DeviceDescriptions& devices = db()->getDeviceDescriptions(); 
  update( const_cast<SiStripConfigDb::DeviceDescriptions&>(devices) );
  if ( doUploadConf() ) { 
    edm::LogVerbatim(mlDqmClient_) 
      << "[OptoScanHistosUsingDb::" << __func__ << "]"
      << " Uploading LLD settings to DB...";
    db()->uploadDeviceDescriptions(true); 
    edm::LogVerbatim(mlDqmClient_) 
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
    const SiStripConfigDb::DeviceAddress& addr = db()->deviceAddress(*desc);
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
      Analyses::const_iterator iter = data().find( fec_key );
      if ( iter != data().end() ) {

	if ( !iter->second->isValid() ) { continue; }

	OptoScanAnalysis* anal = dynamic_cast<OptoScanAnalysis*>( iter->second );
	if ( !anal ) { continue; }
	
	uint16_t gain = anal->gain();
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
	desc->setBias( ichan, anal->bias()[gain] );
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

// -----------------------------------------------------------------------------
/** */
void OptoScanHistosUsingDb::create( SiStripConfigDb::AnalysisDescriptions& desc,
					  Analysis analysis ) {
  
  OptoScanAnalysis* anal = dynamic_cast<OptoScanAnalysis*>( analysis->second );
  if ( !anal ) { return; }
  
  SiStripFecKey key( analysis->first );

  for ( uint16_t iapv = 0; iapv < 2; ++iapv ) {

    // Create description
    OptoScanAnalysisDescription* tmp;
    tmp = new OptoScanAnalysisDescription( anal->gain(),
					   anal->bias()[0],
					   anal->bias()[1],
					   anal->bias()[2],
					   anal->bias()[3],
					   anal->measGain()[0],
					   anal->measGain()[1],
					   anal->measGain()[2],
					   anal->measGain()[3],
					   anal->zeroLight()[0],
					   anal->zeroLight()[1],
					   anal->zeroLight()[2],
					   anal->zeroLight()[3],
					   anal->linkNoise()[0],
					   anal->linkNoise()[1],
					   anal->linkNoise()[2],
					   anal->linkNoise()[3],
					   anal->liftOff()[0],
					   anal->liftOff()[1],
					   anal->liftOff()[2],
					   anal->liftOff()[3],
					   anal->threshold()[0],
					   anal->threshold()[1],
					   anal->threshold()[2],
					   anal->threshold()[3],
					   anal->tickHeight()[0],
					   anal->tickHeight()[1],
					   anal->tickHeight()[2],
					   anal->tickHeight()[3],
					   key.fecCrate(),
					   key.fecSlot(),
					   key.fecRing(),
					   key.ccuAddr(),
					   key.ccuChan(),
					   SiStripFecKey::i2cAddr( key.lldChan(), !iapv ), 
					   db()->dbParams().partition_,
					   db()->dbParams().runNumber_,
					   anal->isValid(),
					   "" );
      
    // Add comments
    typedef std::vector<std::string> Strings;
    Strings errors = anal->getErrorCodes();
    Strings::const_iterator istr = errors.begin();
    Strings::const_iterator jstr = errors.end();
    for ( ; istr != jstr; ++istr ) { tmp->addComments( *istr ); }
    
    // Store description
    desc.push_back( tmp );
    
  }

}

