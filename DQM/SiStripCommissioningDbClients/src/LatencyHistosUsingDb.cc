// Last commit: $Id: LatencyHistosUsingDb.cc,v 1.5 2008/02/20 11:26:12 bainbrid Exp $

#include "DQM/SiStripCommissioningDbClients/interface/LatencyHistosUsingDb.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
LatencyHistosUsingDb::LatencyHistosUsingDb( DQMOldReceiver* mui,
					      const DbParams& params )
  : CommissioningHistosUsingDb( params ),
    LatencyHistograms( mui )
{
  LogTrace(mlDqmClient_) 
    << "[LatencyHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
LatencyHistosUsingDb::LatencyHistosUsingDb( DQMOldReceiver* mui,
					      SiStripConfigDb* const db )
  : CommissioningHistosUsingDb( db ),
    LatencyHistograms( mui )
{
  LogTrace(mlDqmClient_) 
    << "[LatencyHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
LatencyHistosUsingDb::LatencyHistosUsingDb( DQMStore* bei,
					      SiStripConfigDb* const db ) 
  : CommissioningHistosUsingDb( db ),
    LatencyHistograms( bei )
{
  LogTrace(mlDqmClient_) 
    << "[LatencyHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
LatencyHistosUsingDb::~LatencyHistosUsingDb() {
  LogTrace(mlDqmClient_) 
    << "[LatencyHistosUsingDb::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
void LatencyHistosUsingDb::uploadConfigurations() {
  
  if ( !db() ) {
    edm::LogWarning(mlDqmClient_) 
      << "[LatencyHistosUsingDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb interface!"
      << " Aborting upload...";
    return;
  }

  // Update APV descriptions with new Latency settings
  const SiStripConfigDb::DeviceDescriptions& devices = db()->getDeviceDescriptions(); 
  update( const_cast<SiStripConfigDb::DeviceDescriptions&>(devices) );
  if ( doUploadConf() ) { 
    LogTrace(mlDqmClient_) 
      << "[LatencyHistosUsingDb::" << __func__ << "]"
      << " Uploading APV settings to DB...";
    db()->uploadDeviceDescriptions(true); 
    LogTrace(mlDqmClient_) 
      << "[LatencyHistosUsingDb::" << __func__ << "]"
      << " Upload of APV settings to DB finished!";
  } else {
    edm::LogWarning(mlDqmClient_) 
      << "[LatencyHistosUsingDb::" << __func__ << "]"
      << " TEST only! No APV settings will be uploaded to DB...";
  }
  
}

// -----------------------------------------------------------------------------
/** */
void LatencyHistosUsingDb::update( SiStripConfigDb::DeviceDescriptions& devices ) {
  
  // Obtain the latency from the analysis object
  if(!data_.size() || !data_.begin()->second.isValid() ) {
    edm::LogVerbatim(mlDqmClient_) 
      << "[LatencyHistosUsingDb::" << __func__ << "]"
      << " Updated NO Latency settings. No analysis result available !" ;
    return;
  }
  uint16_t latency = uint16_t((data_.begin()->second.maximum()/(-25.))+0.5);
  
  // Iterate through devices and update device descriptions
  uint16_t updated = 0;
  SiStripConfigDb::DeviceDescriptions::iterator idevice;
  for ( idevice = devices.begin(); idevice != devices.end(); idevice++ ) {
    // Check device type
    if ( (*idevice)->getDeviceType() != APV25 ) { continue; }
    // Cast to retrieve appropriate description object
    apvDescription* desc = dynamic_cast<apvDescription*>( *idevice );
    if ( !desc ) { continue; }
    // Retrieve device addresses from device description
    const SiStripConfigDb::DeviceAddress& addr = db()->deviceAddress(*desc);
    // Do it!
    std::stringstream ss;
    ss << "[LatencyHistosUsingDb::" << __func__ << "]"
       << " Updating latency APV settings for crate/FEC/slot/ring/CCU/i2cAddr "
       << addr.fecCrate_ << "/"
       << addr.fecSlot_ << "/"
       << addr.fecRing_ << "/"
       << addr.ccuAddr_ << "/"
       << addr.ccuChan_ << "/"
       << addr.i2cAddr_
       << " from "
       << static_cast<uint16_t>(desc->getLatency());
    desc->setLatency(latency);
    updated++;
    ss << " to "
       << static_cast<uint16_t>(desc->getLatency());
    LogTrace(mlDqmClient_) << ss.str();
  }
  edm::LogVerbatim(mlDqmClient_) 
    << "[LatencyHistosUsingDb::" << __func__ << "] "
    << "Updated Latency settings for " << updated << " devices";
}


// -----------------------------------------------------------------------------
/** */
void LatencyHistosUsingDb::create( SiStripConfigDb::AnalysisDescriptions& desc,
				     Analysis analysis ) {

#ifdef USING_NEW_DATABASE_MODEL
  
  LatencyAnalysis* anal = dynamic_cast<LatencyAnalysis*>( analysis->second );
  if ( !anal ) { return; }
  
  SiStripFecKey fec_key( anal->fecKey() ); //@@ analysis->first
  SiStripFedKey fed_key( anal->fedKey() );
  
  uint16_t latency = static_cast<uint16_t>( ( anal->maximum() / (-25.) ) + 0.5 );
  
  for ( uint16_t iapv = 0; iapv < 2; ++iapv ) {
    
    ApvLatencyAnalysisDescription* tmp;
    tmp = new ApvLatencyAnalysisDescription( latency, 
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

