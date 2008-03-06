// Last commit: $Id: OptoScanHistosUsingDb.cc,v 1.12 2008/02/20 11:26:12 bainbrid Exp $

#include "DQM/SiStripCommissioningDbClients/interface/OptoScanHistosUsingDb.h"
#include "CondFormats/SiStripObjects/interface/OptoScanAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
OptoScanHistosUsingDb::OptoScanHistosUsingDb( DQMOldReceiver* mui,
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
OptoScanHistosUsingDb::OptoScanHistosUsingDb( DQMOldReceiver* mui,
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
OptoScanHistosUsingDb::OptoScanHistosUsingDb( DQMStore* bei,
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
  LogTrace(mlDqmClient_) 
    << "[OptoScanHistosUsingDb::" << __func__ << "]";

  if ( !db() ) {
    edm::LogError(mlDqmClient_) 
      << "[OptoScanHistosUsingDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb interface!"
      << " Aborting upload...";
    return;
  }

  // Retrieve DetInfo
  std::map<uint32_t,DetInfo> info;
  detInfo( info );
  
  // Update LLD descriptions with new bias/gain settings
  const SiStripConfigDb::DeviceDescriptions& devices = db()->getDeviceDescriptions( LASERDRIVER ); 
  update( const_cast<SiStripConfigDb::DeviceDescriptions&>(devices), info );
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
void OptoScanHistosUsingDb::update( SiStripConfigDb::DeviceDescriptions& devices,
				    const DetInfoMap& info ) {

  // Iterate through devices and update device descriptions
  uint16_t updated = 0;
  SiStripConfigDb::DeviceDescriptions::iterator idevice;
  for ( idevice = devices.begin(); idevice != devices.end(); idevice++ ) {
    
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
	if ( !anal ) { 
	  edm::LogError(mlDqmClient_)
	    << "[OptoScanHistosUsingDb::" << __func__ << "]"
	    << " NULL pointer to analysis object!";	
	  continue; 
	}
	
	uint16_t gain = anal->gain();
	std::stringstream ss;
	ss << "[OptoScanHistosUsingDb::" << __func__ << "]"
	   << " Updating LLD gain/bias settings for crate/FEC/slot/ring/CCU/LLD "
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

	// Find DetInfo
	SiStripFecKey key( fec_path, sistrip::CCU_CHAN );
	DetInfoMap::const_iterator iter = info.find( key.key() );
	if ( iter == info.end() ) { 
	  std::stringstream ss;
	  ss << "[OptoScanHistosUsingDb::" << __func__ << "]"
	     << " Unable to find FEC key in DetInfoMap: "; 
	  key.terse(ss);
	  edm::LogWarning(mlDqmClient_) << ss.str();
	  continue;
	}

	// If middle LLD channel and 4-APV module, then do not print warning
	if ( fec_path.channel() == 2 && iter->second.pairs_ == 2 ) { continue; }

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
    << " Updated LLD bias/gain settings for " 
    << updated << " modules";
  

}

// -----------------------------------------------------------------------------
/** */
void OptoScanHistosUsingDb::create( SiStripConfigDb::AnalysisDescriptions& desc,
				    Analysis analysis ) {
  
#ifdef USING_NEW_DATABASE_MODEL

  OptoScanAnalysis* anal = dynamic_cast<OptoScanAnalysis*>( analysis->second );
  if ( !anal ) { return; }
  
  SiStripFecKey fec_key( anal->fecKey() ); 
  SiStripFedKey fed_key( anal->fedKey() );

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
					   fec_key.fecCrate(),
					   fec_key.fecSlot(),
					   fec_key.fecRing(),
					   fec_key.ccuAddr(),
					   fec_key.ccuChan(),
					   SiStripFecKey::i2cAddr( fec_key.lldChan(), !iapv ), 
					   db()->dbParams().partition_,
					   db()->dbParams().runNumber_,
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

