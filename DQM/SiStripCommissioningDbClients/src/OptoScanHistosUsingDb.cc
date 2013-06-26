// Last commit: $Id: OptoScanHistosUsingDb.cc,v 1.22 2010/04/21 14:26:27 dstrom Exp $

#include "DQM/SiStripCommissioningDbClients/interface/OptoScanHistosUsingDb.h"
#include "CondFormats/SiStripObjects/interface/OptoScanAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
OptoScanHistosUsingDb::OptoScanHistosUsingDb( const edm::ParameterSet & pset,
                                              DQMStore* bei,
                                              SiStripConfigDb* const db ) 
  : CommissioningHistograms( pset.getParameter<edm::ParameterSet>("OptoScanParameters"),
                             bei,
                             sistrip::OPTO_SCAN ),
    CommissioningHistosUsingDb( db,
                                sistrip::OPTO_SCAN ),
    OptoScanHistograms( pset.getParameter<edm::ParameterSet>("OptoScanParameters"),
                        bei )
{
  LogTrace(mlDqmClient_) 
    << "[OptoScanHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
  skipGainUpdate_ = this->pset().getParameter<bool>("SkipGainUpdate");
  if (skipGainUpdate_)
    LogTrace(mlDqmClient_)
      << "[OptoScanHistosUsingDb::" << __func__ << "]"
      << " Skipping db update of gain parameters.";
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
  
  // Update LLD descriptions with new bias/gain settings
  SiStripConfigDb::DeviceDescriptionsRange devices = db()->getDeviceDescriptions( LASERDRIVER ); 
  update( devices );
  if ( doUploadConf() ) { 
    edm::LogVerbatim(mlDqmClient_) 
      << "[OptoScanHistosUsingDb::" << __func__ << "]"
      << " Uploading LLD settings to DB...";
    db()->uploadDeviceDescriptions(); 
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
void OptoScanHistosUsingDb::update( SiStripConfigDb::DeviceDescriptionsRange devices ) {
  
  // Iterate through devices and update device descriptions
  uint16_t updated = 0;
  SiStripConfigDb::DeviceDescriptionsV::const_iterator idevice;
  for ( idevice = devices.begin(); idevice != devices.end(); idevice++ ) {
    
    if ( (*idevice)->getDeviceType() != LASERDRIVER ) { continue; }

    // Cast to retrieve appropriate description object
    laserdriverDescription* desc = dynamic_cast<laserdriverDescription*>( *idevice );
    if ( !desc ) { continue; }
    
    // Retrieve device addresses from device description
    const SiStripConfigDb::DeviceAddress& addr = db()->deviceAddress(*desc);
    
    // Iterate through LLD channels
    for ( uint16_t ichan = 0; ichan < sistrip::CHANS_PER_LLD; ichan++ ) {
      
      // Construct key from device description
      SiStripFecKey fec_key( addr.fecCrate_, 
			     addr.fecSlot_, 
			     addr.fecRing_, 
			     addr.ccuAddr_, 
			     addr.ccuChan_,
			     ichan+1 );
      
      // Iterate through all channels and extract LLD settings 
      Analyses::const_iterator iter = data().find( fec_key.key() );
      if ( iter != data().end() ) {

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
	   << " Updating LLD gain/bias settings for crate/crate/FEC/ring/CCU/module/LLD "
	   << fec_key.fecCrate() << "/"
	   << fec_key.fecSlot() << "/"
	   << fec_key.fecRing() << "/"
	   << fec_key.ccuAddr() << "/"
	   << fec_key.ccuChan() << "/"
	   << fec_key.channel() 
	   << " from "
	   << static_cast<uint16_t>( desc->getGain(ichan) ) << "/" 
	   << static_cast<uint16_t>( desc->getBias(ichan) );
        if (!skipGainUpdate_) desc->setGain( ichan, gain );
        desc->setBias( ichan, anal->bias()[gain] );
	updated++;
	ss << " to "
	   << static_cast<uint16_t>(desc->getGain(ichan)) << "/" 
	   << static_cast<uint16_t>(desc->getBias(ichan));
	LogTrace(mlDqmClient_) << ss.str();
	
      } else {
	if ( deviceIsPresent(fec_key) ) { 
	  edm::LogWarning(mlDqmClient_) 
	    << "[OptoScanHistosUsingDb::" << __func__ << "]"
	    << " Unable to find FEC key with params crate/FEC/ring/CCU/module/LLD " 
	    << fec_key.fecCrate() << "/"
	    << fec_key.fecSlot() << "/"
	    << fec_key.fecRing() << "/"
	    << fec_key.ccuAddr() << "/"
	    << fec_key.ccuChan() << "/"
	    << fec_key.channel();
	}
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
void OptoScanHistosUsingDb::create( SiStripConfigDb::AnalysisDescriptionsV& desc,
				    Analysis analysis ) {

  OptoScanAnalysis* anal = dynamic_cast<OptoScanAnalysis*>( analysis->second );
  if ( !anal ) { return; }
  
  SiStripFecKey fec_key( anal->fecKey() ); 
  SiStripFedKey fed_key( anal->fedKey() );

  for ( uint16_t iapv = 0; iapv < 2; ++iapv ) {

    // Create description
    OptoScanAnalysisDescription* tmp;
    tmp = new OptoScanAnalysisDescription( anal->baseSlope()[0],
                                           anal->baseSlope()[1],
                                           anal->baseSlope()[2],
                                           anal->baseSlope()[3],
                                           anal->gain(),
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

}

