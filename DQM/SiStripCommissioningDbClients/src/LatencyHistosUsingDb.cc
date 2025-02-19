// Last commit: $Id: LatencyHistosUsingDb.cc,v 1.22 2009/11/10 14:49:02 lowette Exp $

#include "DQM/SiStripCommissioningDbClients/interface/LatencyHistosUsingDb.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <iostream>

#define MAXFEDCOARSE 15

using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
LatencyHistosUsingDb::LatencyHistosUsingDb( const edm::ParameterSet & pset,
                                            DQMStore* bei,
                                            SiStripConfigDb* const db )
  : CommissioningHistograms( pset.getParameter<edm::ParameterSet>("LatencyParameters"),
                             bei,
                             sistrip::APV_LATENCY ),
    CommissioningHistosUsingDb( db,
                                sistrip::APV_LATENCY ),
    SamplingHistograms( pset.getParameter<edm::ParameterSet>("LatencyParameters"),
                        bei,
                        sistrip::APV_LATENCY )
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

  SiStripConfigDb::DeviceDescriptionsRange devices = db()->getDeviceDescriptions(); 
  SiStripConfigDb::FedDescriptionsRange feds = db()->getFedDescriptions();
  bool upload = update( devices, feds );
  // Check if new PLL settings are valid
  if ( !upload ) {
    edm::LogWarning(mlDqmClient_)
      << "[LatencyHistosUsingDb::" << __func__ << "]"
      << " Found invalid PLL settings (coarse > 15)"
      << " Aborting update to database...";
    return;
  }
  
  if ( doUploadConf() ) { 
    // Update APV descriptions with new Latency settings
    LogTrace(mlDqmClient_) 
      << "[LatencyHistosUsingDb::" << __func__ << "]"
      << " Uploading APV settings to DB...";
    db()->uploadDeviceDescriptions(); 
    LogTrace(mlDqmClient_) 
      << "[LatencyHistosUsingDb::" << __func__ << "]"
      << " Upload of APV settings to DB finished!";
    // Update FED descriptions 
    LogTrace(mlDqmClient_)
      << "[LatencyHistosUsingDb::" << __func__ << "]"
      << " Uploading FED delays to DB...";
    db()->uploadFedDescriptions();
    LogTrace(mlDqmClient_)
      << "[LatencyHistosUsingDb::" << __func__ << "]"
      << " Upload of FED delays to DB finished!";
  } else {
    edm::LogWarning(mlDqmClient_) 
      << "[LatencyHistosUsingDb::" << __func__ << "]"
      << " TEST only! No APV settings will be uploaded to DB...";
  }

}

// -----------------------------------------------------------------------------
/** */
bool LatencyHistosUsingDb::update( SiStripConfigDb::DeviceDescriptionsRange devices, 
				   SiStripConfigDb::FedDescriptionsRange feds ) {
  
  // Obtain the latency from the analysis object
  if(!data().size() || !data().begin()->second->isValid() ) {
    edm::LogVerbatim(mlDqmClient_) 
      << "[LatencyHistosUsingDb::" << __func__ << "]"
      << " Updated NO Latency settings. No analysis result available !" ;
    return false;
  }

  // Compute the minimum coarse delay
  uint16_t minCoarseDelay = 256;
  SiStripConfigDb::DeviceDescriptionsV::const_iterator idevice;
  for ( idevice = devices.begin(); idevice != devices.end(); idevice++ ) {
    // Check device type
    if ( (*idevice)->getDeviceType() == PLL ) {
      // Cast to retrieve appropriate description object
      pllDescription* desc = dynamic_cast<pllDescription*>( *idevice );
      if ( desc ) { 
/*
        // add 1 to aim at 1 and not 0 (just to avoid a special 0 value for security)
        int delayCoarse = desc->getDelayCoarse() - 1;
        delayCoarse = delayCoarse < 0 ? 0 : delayCoarse;
        minCoarseDelay = minCoarseDelay < delayCoarse ? minCoarseDelay : delayCoarse;
*/
        int delayCoarse = desc->getDelayCoarse();
        minCoarseDelay = minCoarseDelay < delayCoarse ? minCoarseDelay : delayCoarse;
      }
    }
  }

  // Compute latency and PLL shift from the sampling measurement
  SamplingAnalysis* anal = NULL;
  for( CommissioningHistograms::Analysis it = data().begin(); it!=data().end();++it) {
    if(dynamic_cast<SamplingAnalysis*>( it->second ) && 
       dynamic_cast<SamplingAnalysis*>( it->second )->granularity()==sistrip::TRACKER)
      anal = dynamic_cast<SamplingAnalysis*>( it->second );
  }
  if(!anal) return false;
  uint16_t globalLatency = uint16_t(ceil(anal->maximum()/(-25.)));
  float globalShift = anal->maximum()-(globalLatency*(-25));

  // Compute latency and PLL shift per partition... this is an option
  uint16_t latency = globalLatency;
  float shift[5] = {0.};
  for( CommissioningHistograms::Analysis it = data().begin(); it!=data().end();++it) {
    if(dynamic_cast<SamplingAnalysis*>( it->second ) &&
       dynamic_cast<SamplingAnalysis*>( it->second )->granularity()==sistrip::PARTITION       )
      anal = dynamic_cast<SamplingAnalysis*>( it->second );
      latency = uint16_t(ceil(anal->maximum()/(-25.)))>latency ? uint16_t(ceil(anal->maximum()/(-25.))) : latency;
  }
  for( CommissioningHistograms::Analysis it = data().begin(); it!=data().end();++it) {
    if(dynamic_cast<SamplingAnalysis*>( it->second ) &&
       dynamic_cast<SamplingAnalysis*>( it->second )->granularity()==sistrip::PARTITION       )
      anal = dynamic_cast<SamplingAnalysis*>( it->second );
      shift[SiStripFecKey(anal->fecKey()).fecCrate()] = anal->maximum()-(latency*(-25));
  }
  if(!perPartition_) {
    latency = globalLatency;
    for(int i=0;i<5;i++) shift[i] = globalShift;
  }

  // Take into account the minimum coarse delay to bring the coarse delay down
  // the same quantity is subtracted to the coarse delay of each APV 
  latency -= minCoarseDelay;
  
  // Iterate through devices and update device descriptions
  uint16_t updatedAPV = 0;
  uint16_t updatedPLL = 0;
  std::vector<SiStripFecKey> invalid;
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
    ss << " to "
       << static_cast<uint16_t>(desc->getLatency());
    LogTrace(mlDqmClient_) << ss.str();
    updatedAPV++;
  }

  // Change also the PLL delay
  for ( idevice = devices.begin(); idevice != devices.end(); idevice++ ) {
    // Check device type
    if ( (*idevice)->getDeviceType() != PLL ) { continue; }
    // Cast to retrieve appropriate description object
    pllDescription* desc = dynamic_cast<pllDescription*>( *idevice );
    if ( !desc ) { continue; }
    if ( desc->getDelayCoarse() >= 15 ) { continue; }
    // Retrieve device addresses from device description
    const SiStripConfigDb::DeviceAddress& addr = db()->deviceAddress(*desc);
    // Construct key from device description
    uint32_t fec_key = SiStripFecKey( addr.fecCrate_,
                                      addr.fecSlot_,
                                      addr.fecRing_,
                                      addr.ccuAddr_,
                                      addr.ccuChan_,
                                      0 ).key();
    SiStripFecKey fec_path = SiStripFecKey( fec_key );    
    // Do it!
    float delay = desc->getDelayCoarse()*25+desc->getDelayFine()*25./24. + shift[addr.fecCrate_];
    int delayCoarse = int(delay/25);
    int delayFine   = int(round((delay-25*delayCoarse)*24./25.));
    if(delayFine==24) { delayFine=0; ++delayCoarse; }
    delayCoarse -= minCoarseDelay;
    //  maximum coarse setting
    if ( delayCoarse > 15 ) { invalid.push_back(fec_key); delayCoarse = sistrip::invalid_; }
    // Update PLL settings
    if ( delayCoarse != sistrip::invalid_ &&
         delayFine != sistrip::invalid_ ) {
      std::stringstream ss;
      ss << "[LatencyHistosUsingDb::" << __func__ << "]"
	 << " Updating coarse/fine PLL settings"
	 << " for Crate/FEC/slot/ring/CCU "
	 << fec_path.fecCrate() << "/"
	 << fec_path.fecSlot() << "/"
	 << fec_path.fecRing() << "/"
	 << fec_path.ccuAddr() << "/"
	 << fec_path.ccuChan()
	 << " from "
	 << static_cast<uint16_t>( desc->getDelayCoarse() ) << "/"
	 << static_cast<uint16_t>( desc->getDelayFine() );
      desc->setDelayCoarse(delayCoarse);
      desc->setDelayFine(delayFine);
      updatedPLL++;
      ss << " to "
	 << static_cast<uint16_t>( desc->getDelayCoarse() ) << "/"
	 << static_cast<uint16_t>( desc->getDelayFine() );
      LogTrace(mlDqmClient_) << ss.str();
    }
  }
  
  // Retrieve FED ids from cabling
  std::vector<uint16_t> ids = cabling()->feds() ;

  // loop over the FED ids to determine min and max values of coarse delay
  uint16_t minDelay = 256;
  uint16_t maxDelay = 0;
  uint16_t fedDelayCoarse = 0;
  for ( SiStripConfigDb::FedDescriptionsV::const_iterator ifed = feds.begin(); ifed != feds.end(); ifed++ ) {
    // If FED id not found in list (from cabling), then continue
    if ( find( ids.begin(), ids.end(), (*ifed)->getFedId() ) == ids.end() ) { continue; }
    const std::vector<FedChannelConnection>& conns = cabling()->connections((*ifed)->getFedId());
    // loop over the connections for that FED
    for ( std::vector<FedChannelConnection>::const_iterator iconn = conns.begin(); iconn != conns.end(); iconn++ ) {
      // check that this is a tracker module
      if(DetId(iconn->detId()).det()!=DetId::Tracker) continue;
      // build the Fed9UAddress for that channel. Used to update the description.
      Fed9U::Fed9UAddress fedChannel = Fed9U::Fed9UAddress(iconn->fedCh());
      // retreive the current value for the delays
      fedDelayCoarse = (*ifed)->getCoarseDelay(fedChannel);
      // update min and max
      minDelay = minDelay<fedDelayCoarse ? minDelay : fedDelayCoarse;
      maxDelay = maxDelay>fedDelayCoarse ? maxDelay : fedDelayCoarse;
    }
  }

  // compute the FED coarse global offset
  int offset = (10-minDelay)*25;  // try to ensure 10BX room for later fine delay scan
  if(maxDelay+(offset/25)>MAXFEDCOARSE) offset = (MAXFEDCOARSE-maxDelay)*25; // otherwise, take the largest possible

  // loop over the FED ids
  for ( SiStripConfigDb::FedDescriptionsV::const_iterator ifed = feds.begin(); ifed != feds.end(); ifed++ ) {
    // If FED id not found in list (from cabling), then continue
    if ( find( ids.begin(), ids.end(), (*ifed)->getFedId() ) == ids.end() ) { continue; }
    const std::vector<FedChannelConnection>& conns = cabling()->connections((*ifed)->getFedId());
    // loop over the connections for that FED
    for ( std::vector<FedChannelConnection>::const_iterator iconn = conns.begin(); iconn != conns.end(); iconn++ ) {
      // check that this is a tracker module
      if(DetId(iconn->detId()).det()!=DetId::Tracker) continue;
      // build the Fed9UAddress for that channel. Used to update the description.
      Fed9U::Fed9UAddress fedChannel = Fed9U::Fed9UAddress(iconn->fedCh());
      // retreive the current value for the delays
      int fedDelayCoarse = (*ifed)->getCoarseDelay(fedChannel);
      int fedDelayFine = (*ifed)->getFineDelay(fedChannel);
      // compute the FED delay
      // this is done by substracting the best (PLL) delay to the present value (from the db)
      int fedDelay = int(fedDelayCoarse*25. - fedDelayFine*24./25. - round(shift[iconn->fecCrate()]) + offset);
      fedDelayCoarse = (fedDelay/25)+1;
      fedDelayFine = fedDelayCoarse*25-fedDelay;
      if(fedDelayFine==25) { fedDelayFine = 0; --fedDelayCoarse; }
      // update the FED delay
      std::stringstream ss;
      ss << "[LatencyHistosUsingDb::" << __func__ << "]"
         << " Updating the FED delay"
         << " for loop FED id/ch "
         << (*ifed)->getFedId() << "/" << iconn->fedCh()
         << " from "
         << (*ifed)->getCoarseDelay( fedChannel) << "/" << (*ifed)->getFineDelay( fedChannel)
         << " to ";
      (*ifed)->setDelay(fedChannel, fedDelayCoarse, fedDelayFine);
      ss << (*ifed)->getCoarseDelay(fedChannel) << "/" << (*ifed)->getFineDelay( fedChannel);
      LogTrace(mlDqmClient_) << ss.str();
    }
  }

  // Summary output
  edm::LogVerbatim(mlDqmClient_)
    << "[LatencyHistosUsingDb::" << __func__ << "]"
    << " Updated FED delays for " << ids.size() << " FEDs!";
  
  // Check if invalid settings were found
  if ( !invalid.empty() ) {
    std::stringstream ss;
    ss << "[LatencyHistosUsingDb::" << __func__ << "]"
       << " Found PLL coarse setting of 15"
       << " (not allowed!) for following channels"
       << " (Crate/FEC/slot/ring/CCU/LLD): ";
    std::vector<SiStripFecKey>::iterator ikey = invalid.begin();
    std::vector<SiStripFecKey>::iterator jkey = invalid.end();
    for ( ; ikey != jkey; ++ikey ) {
      ss << ikey->fecCrate() << "/"
         << ikey->fecSlot() << "/"
         << ikey->fecRing() << "/"
         << ikey->ccuAddr() << "/"
         << ikey->ccuChan() << ", ";
    }
    edm::LogWarning(mlDqmClient_) << ss.str();
    return false;
  }

  // Summary output
  edm::LogVerbatim(mlDqmClient_) 
    << "[LatencyHistosUsingDb::" << __func__ << "] "
    << "Updated settings for " << updatedAPV << " APV devices and " << updatedPLL << " PLL devices.";
  return true;
}

// -----------------------------------------------------------------------------
/** */
void LatencyHistosUsingDb::create( SiStripConfigDb::AnalysisDescriptionsV& desc,
				   Analysis analysis ) {

  SamplingAnalysis* anal = dynamic_cast<SamplingAnalysis*>( analysis->second );
  if ( !anal ) { return; }
  
  SiStripFecKey fec_key( anal->fecKey() ); //@@ analysis->first
  SiStripFedKey fed_key( anal->fedKey() );
  
  uint16_t latency = static_cast<uint16_t>( ( anal->maximum() / (-25.) ) + 0.5 );

  ApvLatencyAnalysisDescription* tmp;
  tmp = new ApvLatencyAnalysisDescription( latency, 
					   0,
					   0,
					   0,
					   0,
					   0,
					   0, 
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

void LatencyHistosUsingDb::configure( const edm::ParameterSet& pset, const edm::EventSetup& es)
{
  perPartition_ = this->pset().getParameter<bool>("OptimizePerPartition");
}

