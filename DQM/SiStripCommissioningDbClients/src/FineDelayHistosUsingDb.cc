// Last commit: $Id: FineDelayHistosUsingDb.cc,v 1.8 2008/03/08 17:24:52 delaer Exp $

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DQM/SiStripCommissioningDbClients/interface/FineDelayHistosUsingDb.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include <Geometry/CommonTopologies/interface/Topology.h>
#include <CondFormats/DataRecord/interface/SiStripFedCablingRcd.h>
#include <CondFormats/SiStripObjects/interface/SiStripFedCabling.h>
#include <CondFormats/SiStripObjects/interface/FedChannelConnection.h>

#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
FineDelayHistosUsingDb::FineDelayHistosUsingDb( DQMOldReceiver* mui,
						const DbParams& params )
  : CommissioningHistosUsingDb( params ),
    SamplingHistograms( mui, FINE_DELAY ),
    tracker_(0)
{
  LogTrace(mlDqmClient_) 
    << "[FineDelayHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
  delays_.clear();
}

// -----------------------------------------------------------------------------
/** */
FineDelayHistosUsingDb::FineDelayHistosUsingDb( DQMOldReceiver* mui,
						SiStripConfigDb* const db ) 
  : CommissioningHistosUsingDb( db ),
    SamplingHistograms( mui, FINE_DELAY ),
    tracker_(0)
{
  LogTrace(mlDqmClient_) 
    << "[FineDelayHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
  delays_.clear();
}

// -----------------------------------------------------------------------------
/** */
FineDelayHistosUsingDb::FineDelayHistosUsingDb( DQMStore* bei,
						SiStripConfigDb* const db ) 
  : CommissioningHistosUsingDb( db ),
    SamplingHistograms( bei, FINE_DELAY ),
    tracker_(0)
{
  LogTrace(mlDqmClient_) 
    << "[FineDelayHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
  delays_.clear();
}

// -----------------------------------------------------------------------------
/** */
FineDelayHistosUsingDb::~FineDelayHistosUsingDb() {
  LogTrace(mlDqmClient_) 
    << "[FineDelayHistosUsingDb::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
void FineDelayHistosUsingDb::configure(const edm::ParameterSet& pset, const edm::EventSetup& setup) {
  // get geometry
  edm::ESHandle<TrackerGeometry> estracker;
  setup.get<TrackerDigiGeometryRecord>().get(estracker);
  tracker_=&(* estracker);
  SamplingHistograms::configure(pset,setup);
}

// -----------------------------------------------------------------------------
/** */
void FineDelayHistosUsingDb::uploadConfigurations() {
  
  if ( !db() ) {
    edm::LogWarning(mlDqmClient_) 
      << "[FineDelayHistosUsingDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb interface!"
      << " Aborting upload...";
    return;
  }
  
    // Retrieve and update PLL device descriptions
    const SiStripConfigDb::DeviceDescriptions& devices = db()->getDeviceDescriptions( PLL ); 
    bool upload = update( const_cast<SiStripConfigDb::DeviceDescriptions&>(devices) );
    
    // Check if new PLL settings are valid 
    if ( !upload ) {
      edm::LogWarning(mlDqmClient_) 
	<< "[FineDelayHistosUsingDb::" << __func__ << "]"
	<< " Found invalid PLL settings (coarse > 15)"
	<< " Aborting update to database...";
      return;
    }
    
    // Upload PLL device descriptions
    if ( doUploadConf() ) { 
      LogTrace(mlDqmClient_) 
	<< "[FineDelayHistosUsingDb::" << __func__ << "]"
	<< " Uploading PLL settings to DB...";
      db()->uploadDeviceDescriptions(true); 
      LogTrace(mlDqmClient_) 
	<< "[FineDelayHistosUsingDb::" << __func__ << "]"
	<< " Upload of PLL settings to DB finished!";
    } else {
      edm::LogWarning(mlDqmClient_) 
	<< "[FineDelayHistosUsingDb::" << __func__ << "]"
	<< " TEST only! No PLL settings will be uploaded to DB...";
    }

    // Update FED descriptions with new ticker thresholds
    const SiStripConfigDb::FedDescriptions& feds = db()->getFedDescriptions(); 
    update( const_cast<SiStripConfigDb::FedDescriptions&>(feds) );

    // Update FED descriptions with new ticker thresholds
    if ( doUploadConf() ) { 
      LogTrace(mlDqmClient_) 
	<< "[FineDelayHistosUsingDb::" << __func__ << "]"
	<< " Uploading FED ticker thresholds to DB...";
      db()->uploadFedDescriptions(true); 
      LogTrace(mlDqmClient_) 
	<< "[FineDelayHistosUsingDb::" << __func__ << "]"
	<< " Upload of FED ticker thresholds to DB finished!";
    } else {
      edm::LogWarning(mlDqmClient_) 
	<< "[FineDelayHistosUsingDb::" << __func__ << "]"
	<< " TEST only! No FED ticker thresholds will be uploaded to DB...";
    }

}

void FineDelayHistosUsingDb::computeDelays() {
  // do nothing if delays_ map is already filled
  if(delays_.size()>0) return;

  // the point from which track should originate
  float x = 0.;
  float y = 0.;
  float z = 0.;
  GlobalPoint referenceP_ = GlobalPoint(x,y,z);
  const double c = 30; // cm/ns
  
  // the reference parameters (best delay in ns, initial Latency)
  float bestDelay_ = 0.;
  if(data().size()) {
    Analyses::const_iterator iter = data().begin();
    bestDelay_ = dynamic_cast<SamplingAnalysis*>(iter->second)->maximum();
  }

  // Retrieve FED ids from cabling
  std::vector<uint16_t> ids = cabling()->feds() ;
  
  // loop over the FED ids
  for (std::vector<uint16_t>::const_iterator ifed = ids.begin(); ifed != ids.end(); ++ifed) {
    const std::vector<FedChannelConnection>& conns = cabling()->connections(*ifed);
    // loop over the connections for that FED 
    for ( std::vector<FedChannelConnection>::const_iterator iconn = conns.begin(); iconn != conns.end(); iconn++ ) {
      // check that this is a tracker module
      if(DetId(iconn->detId()).det()!=DetId::Tracker) continue;
      // retrieve the position of that module in the tracker using the geometry
      // and use it to compute the distance to the reference point set in the configuration
      if(tracker_) {
        float dist = tracker_->idToDetUnit(DetId(iconn->detId()))->toLocal(referenceP_).mag(); 
        float tof  = dist/c ;
        // compute the PLL delay shift for the module as delay + tof 
        int delay = int(round(bestDelay_+tof));
        // store that in the map
        delays_[SiStripFecKey( iconn->fecCrate(),
                               iconn->fecSlot(),
                               iconn->fecRing(),
                               iconn->ccuAddr(),
                               iconn->ccuChan(), 0 ).key()] = delay;
      } else {
        edm::LogError(mlDqmClient_)
           << "[FineDelayHistosUsingDb::" << __func__ << "]"
           << " Tracker geometry not initialized. Impossible to compute the delays.";
      }
    }
  }
}

// -----------------------------------------------------------------------------
/** */
bool FineDelayHistosUsingDb::update( SiStripConfigDb::DeviceDescriptions& devices ) {

  // do the core computation of delays per FED connection
  computeDelays();

  // Iterate through devices and update device descriptions
  uint16_t updated = 0;
  std::vector<SiStripFecKey> invalid;
  SiStripConfigDb::DeviceDescriptions::iterator idevice;
  for ( idevice = devices.begin(); idevice != devices.end(); idevice++ ) {
    
    // Check device type
    if ( (*idevice)->getDeviceType() != PLL ) { continue; }
    
    // Cast to retrieve appropriate description object
    pllDescription* desc = dynamic_cast<pllDescription*>( *idevice ); 
    if ( !desc ) { continue; }
    
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
    
    // extract the delay from the map
    int delay = delays_[fec_key];
    int delayCoarse = delay/25;
    int delayFine   = int(round((delay-25*delayCoarse)*24./25.));
    //  maximum coarse setting
    if ( delayCoarse > 15 ) { invalid.push_back(fec_key); delayCoarse = sistrip::invalid_; }
		    
    // Update PLL settings
    if ( delayCoarse != sistrip::invalid_ && 
	 delayFine != sistrip::invalid_ ) { 
      
      std::stringstream ss;
      ss << "[FineDelayHistosUsingDb::" << __func__ << "]"
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
      updated++;
      ss << " to "
	 << static_cast<uint16_t>( desc->getDelayCoarse() ) << "/" 
	 << static_cast<uint16_t>( desc->getDelayFine() );
      LogTrace(mlDqmClient_) << ss.str();

    } else {
      LogTrace(mlDqmClient_) 
	<< "[FineDelayHistosUsingDb::" << __func__ << "]"
	<< " Unexpected PLL delay settings for Crate/FEC/slot/ring/CCU " 
	<< fec_path.fecCrate() << "/"
	<< fec_path.fecSlot() << "/"
	<< fec_path.fecRing() << "/"
	<< fec_path.ccuAddr() << "/"
	<< fec_path.ccuChan();
    }

  }

  // Check if invalid settings were found
  if ( !invalid.empty() ) {
    std::stringstream ss;
    ss << "[FineDelayHistosUsingDb::" << __func__ << "]"
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
  
  edm::LogVerbatim(mlDqmClient_) 
    << "[FineDelayHistosUsingDb::" << __func__ << "]"
    << " Updated PLL settings for " 
    << updated << " modules";
  return true;
}

// -----------------------------------------------------------------------------
/** */
void FineDelayHistosUsingDb::update( SiStripConfigDb::FedDescriptions& feds ) {

  // do the core computation of delays per FED connection
  computeDelays();

  // Retrieve FED ids from cabling
  std::vector<uint16_t> ids = cabling()->feds() ;
  
  // loop over the FED ids
  for ( SiStripConfigDb::FedDescriptions::iterator ifed = feds.begin(); ifed != feds.end(); ifed++ ) {
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
      int fedDelay = int(fedDelayCoarse*25. - fedDelayFine*24./25.);
      // extract the delay from the map
      int delay = delays_[SiStripFecKey( iconn->fecCrate(),
                                         iconn->fecSlot(),
                                         iconn->fecRing(),
                                         iconn->ccuAddr(),
                                         iconn->ccuChan(), 0 ).key()];
      // compute the FED delay
      // this is done by substracting the best (PLL) delay to the present value (from the db)
      fedDelay -= delay;
      fedDelayCoarse = (fedDelay/25)+1;
      fedDelayFine = fedDelayCoarse*25-fedDelay;
      // update the FED delay
      std::stringstream ss;
      ss << "[FineDelayHistosUsingDb::" << __func__ << "]"
         << " Updating the FED delay"
         << " for loop FED id/ch "
         << (*ifed)->getFedId() << "/" << iconn->fedCh()
         << " from "
         << (*ifed)->getCoarseDelay( fedChannel) << "/" << (*ifed)->getFineDelay( fedChannel)
         << " to ";
      (*ifed)->setDelay(fedChannel, fedDelayCoarse, fedDelayFine);
      ss << (*ifed)->getCoarseDelay(fedChannel) << "/" << (*ifed)->getFineDelay( fedChannel) << std::endl;
      LogTrace(mlDqmClient_) << ss.str();
    }
  }

  edm::LogVerbatim(mlDqmClient_)
      << "[FineDelayHistosUsingDb::" << __func__ << "]"
      << " Updated FED delay for " << ids.size() << " FEDs!";

}

// -----------------------------------------------------------------------------
/** */
void FineDelayHistosUsingDb::create( SiStripConfigDb::AnalysisDescriptions& desc,
				     Analysis analysis ) {

#ifdef USING_NEW_DATABASE_MODEL

  SamplingAnalysis* anal = dynamic_cast<SamplingAnalysis*>( analysis->second );
  if ( !anal ) { return; }
  
  SiStripFecKey fec_key( anal->fecKey() ); //@@ analysis->first
  SiStripFedKey fed_key( anal->fedKey() );

    FineDelayAnalysisDescription* tmp;
    tmp = new FineDelayAnalysisDescription( anal->maximum(),
					    anal->error(),
					    0,
					    0,
					    0,
					    0,
					    0,
					    0, 
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
    
#endif
  
}
