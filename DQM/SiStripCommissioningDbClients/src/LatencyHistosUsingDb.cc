// Last commit: $Id: LatencyHistosUsingDb.cc,v 1.2 2007/12/11 17:11:12 delaer Exp $

#include "DQM/SiStripCommissioningDbClients/interface/LatencyHistosUsingDb.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
LatencyHistosUsingDb::LatencyHistosUsingDb( MonitorUserInterface* mui,
					      const DbParams& params )
  : LatencyHistograms( mui ),
    CommissioningHistosUsingDb( params )
{
  LogTrace(mlDqmClient_) 
    << "[LatencyHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
LatencyHistosUsingDb::LatencyHistosUsingDb( MonitorUserInterface* mui,
					      SiStripConfigDb* const db )
  : LatencyHistograms( mui ),
    CommissioningHistosUsingDb( db )
{
  LogTrace(mlDqmClient_) 
    << "[LatencyHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
LatencyHistosUsingDb::LatencyHistosUsingDb( DaqMonitorBEInterface* bei,
					      SiStripConfigDb* const db ) 
  : LatencyHistograms( bei ),
    CommissioningHistosUsingDb( db )
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
void LatencyHistosUsingDb::create( SiStripConfigDb::AnalysisDescriptions& desc ) {

  edm::LogVerbatim(mlDqmClient_) 
    << "[ApvTimingHistosUsingDb::" << __func__ << "]"
    << " Creating TimingAnalysisDescriptions...";
  
  // Clear descriptions container
  desc.clear();
  
  // Iterate through analysis objects and create analysis descriptions
  typedef std::map<uint32_t,LatencyAnalysis> Analyses; 
  Analyses::iterator ianal  = data_.begin();
  Analyses::iterator janal  = data_.end();
  for ( ; ianal != janal; ++ianal ) {

    LatencyAnalysis* anal = &(ianal->second);
    if ( !anal ) { continue; }

    SiStripFecKey key( ianal->first );

    uint16_t latency = static_cast<uint16_t>( ( data_.begin()->second.maximum() / (-25.) ) + 0.5 );

    for ( uint16_t iapv = 0; iapv < 2; ++iapv ) {

      // Create description
      ApvLatencyAnalysisDescription* tmp;
      tmp = new ApvLatencyAnalysisDescription( latency, 
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

//   std::stringstream sss;
//   SiStripConfigDb::AnalysisDescriptions::iterator ii = desc.begin();
//   SiStripConfigDb::AnalysisDescriptions::iterator jj = desc.end();
//   for ( ; ii != jj; ++ii ) { 
//     ApvLatencyAnalysisDescription* tmp = dynamic_cast<ApvLatencyAnalysisDescription*>( *ii );
//     if ( tmp ) { sss << tmp->toString(); }
//   }
//   edm::LogVerbatim(mlDqmClient_) 
//     << "[FastFedCablingHistosUsingDb::" << __func__ << "]"
//     << " Analysis descriptions:" << std::endl << sss.str(); 

}

