// Last commit: $Id: FastFedCablingHistosUsingDb.cc,v 1.3 2007/11/20 22:40:53 bainbrid Exp $

#include "DQM/SiStripCommissioningDbClients/interface/FastFedCablingHistosUsingDb.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
FastFedCablingHistosUsingDb::FastFedCablingHistosUsingDb( MonitorUserInterface* mui,
							  const DbParams& params )
  : FastFedCablingHistograms( mui ),
    CommissioningHistosUsingDb( params )
{
  LogTrace(mlDqmClient_)
    << "[FastFedCablingHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
FastFedCablingHistosUsingDb::FastFedCablingHistosUsingDb( MonitorUserInterface* mui,
							  SiStripConfigDb* const db ) 
  : FastFedCablingHistograms( mui ),
    CommissioningHistosUsingDb( db )
{
  LogTrace(mlDqmClient_)
    << "[FastFedCablingHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
FastFedCablingHistosUsingDb::FastFedCablingHistosUsingDb( DaqMonitorBEInterface* bei,
							  SiStripConfigDb* const db ) 
  : FastFedCablingHistograms( bei ),
    CommissioningHistosUsingDb( db )
{
  LogTrace(mlDqmClient_) 
    << "[FastFedCablingHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
FastFedCablingHistosUsingDb::~FastFedCablingHistosUsingDb() {
  LogTrace(mlDqmClient_)
    << "[FastFedCablingHistosUsingDb::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
void FastFedCablingHistosUsingDb::uploadToConfigDb() {
  
  if ( !db_ ) {
    edm::LogWarning(mlDqmClient_) 
      << "[FastFedCablingHistosUsingDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb interface!"
      << " Aborting upload...";
    return;
  }

  // Retrieve descriptions for DCU id and DetId 
  SiStripConfigDb::DeviceDescriptions dcus = db_->getDeviceDescriptions( DCU ); 
  SiStripConfigDb::DcuDetIdMap detids = db_->getDcuDetIdMap(); 
  
  // Update FED connection descriptions
  const SiStripConfigDb::FedConnections& conns = db_->getFedConnections(); 
  update( const_cast<SiStripConfigDb::FedConnections&>(conns), dcus, detids );
  if ( !test_ ) { 
    edm::LogVerbatim(mlDqmClient_) 
      << "[FastFedCablingHistosUsingDb::" << __func__ << "]"
      << " Uploading FED connections to DB...";
    db_->uploadFedConnections(true); 
    edm::LogVerbatim(mlDqmClient_) 
      << "[FastFedCablingHistosUsingDb::" << __func__ << "]"
      << " Completed database upload of " << conns.size() 
      << " ConnectionDescriptions!";
  } else {
    edm::LogWarning(mlDqmClient_) 
      << "[FastFedCablingHistosUsingDb::" << __func__ << "]"
      << " TEST only! No FED connections will be uploaded to DB...";
  }
  
  // Update FED descriptions with enabled/disabled channels
  const SiStripConfigDb::FedDescriptions& feds = db_->getFedDescriptions(); 
  update( const_cast<SiStripConfigDb::FedDescriptions&>(feds) );
  if ( !test_ ) { 
    edm::LogVerbatim(mlDqmClient_) 
      << "[FastFedCablingHistosUsingDb::" << __func__ << "]"
      << " Uploading FED descriptions to DB...";
    db_->uploadFedDescriptions(true); 
    edm::LogVerbatim(mlDqmClient_) 
      << "[FastFedCablingHistosUsingDb::" << __func__ << "]"
      << " Completed database upload of " << feds.size()
      << " Fed9UDescriptions (with connected channels enabled)!";
  } else {
    edm::LogWarning(mlDqmClient_) 
      << "[FastFedCablingHistosUsingDb::" << __func__ << "]"
      << " TEST only! No FED descriptions will be uploaded to DB...";
  }
  
}

// -----------------------------------------------------------------------------
/** */
void FastFedCablingHistosUsingDb::update( SiStripConfigDb::FedConnections& conns,
					  const SiStripConfigDb::DeviceDescriptions& dcus, 
					  const SiStripConfigDb::DcuDetIdMap& detids ) {

  // Clear any connections retrieved from database
  if ( !conns.empty() ) {
    edm::LogWarning(mlDqmClient_)
      << "[FastFedCablingHistosUsingDb::" << __func__ << "]"
      << " Clearing existing FED channel connections!";
    conns.clear();
  }
  
  // Counter for unconnected channels
  uint16_t unconnected = 0;

  // Update FED-FEC mapping in base class, based on analysis results
  map<uint32_t,FastFedCablingAnalysis*>::const_iterator ianal;
  for ( ianal = data_.begin(); ianal != data_.end(); ianal++ ) {
    
    if ( !ianal->second->isValid() ) { continue; }
    
    // Retrieve DCU id matching id from histogram
    bool found = false;
    SiStripConfigDb::DeviceDescriptions::const_iterator idcu = dcus.begin();
    while ( !found && idcu != dcus.end() ) {
      
      // Check if DCU hard id matches that provided by analysis
      dcuDescription* dcu = dynamic_cast<dcuDescription*>( *idcu );
      if ( dcu ) { 
	if ( dcu->getDcuType() == "FEH" ) { 
	  if ( dcu->getDcuHardId() == ianal->second->dcuId() ) {
	    
	    // Label as "found"
	    found = true; 
	    
	    // Retrieve I2C addresses
	    const SiStripConfigDb::DeviceAddress& addr = db_->deviceAddress(*dcu);
	
	    // Create connection object and set member data 
#ifdef USING_NEW_DATABASE_MODEL	
	    ConnectionDescription* conn = new ConnectionDescription();
	    conn->setFedId( ianal->second->fedKey().fedId() );
	    conn->setFedChannel( ianal->second->fedKey().fedChannel() );
	    conn->setFecHardwareId( "" ); //@@
	    conn->setCrateSlot( addr.fecCrate_ - sistrip::FEC_CRATE_OFFSET );
	    conn->setFecSlot( addr.fecSlot_ );
	    conn->setRingSlot( addr.fecRing_ - sistrip::FEC_RING_OFFSET );
	    conn->setCcuAddress( addr.ccuAddr_ );
	    conn->setI2cChannel( addr.ccuChan_ );
	    conn->setApvAddress( SiStripFecKey::i2cAddr(ianal->second->lldCh(),true) );
	    conn->setDcuHardId( dcu->getDcuHardId() );
	    //@@ conn->setDetId( sistrip::invalid32_ );
	    //@@ conn->setApvPairs( sistrip::invalid_ );
	    //@@ conn->setFiberLength( sistrip::invalid_ );
#else
	    FedChannelConnectionDescription* conn = new FedChannelConnectionDescription(); 
	    conn->setFedId( ianal->second->fedKey().fedId() );
	    conn->setFedChannel( ianal->second->fedKey().fedChannel() );
	    conn->setFecSupervisor( "" );
	    conn->setFecSupervisorIP( "" );
	    conn->setFecInstance( addr.fecCrate_ - sistrip::FEC_CRATE_OFFSET );
	    conn->setSlot( addr.fecSlot_ );
	    conn->setRing( addr.fecRing_ - sistrip::FEC_RING_OFFSET );
	    conn->setCcu( addr.ccuAddr_ );
	    conn->setI2c( addr.ccuChan_ );
	    conn->setApv( SiStripFecKey::i2cAddr(ianal->second->lldCh(),true) );
	    conn->setDcuHardId( dcu->getDcuHardId() );
	    conn->setDetId( sistrip::invalid32_ );
	    conn->setApvPairs( sistrip::invalid_ ); 
	    conn->setFiberLength( sistrip::invalid_ );
#endif

	    // Retrieve DetId from DB and set member data 
	    SiStripConfigDb::DcuDetIdMap::const_iterator idet = detids.find( conn->getDcuHardId() );
	    if ( idet != detids.end() ) { 
#ifndef USING_NEW_DATABASE_MODEL
	      conn->setDetId( idet->second->getDetId() );
	      conn->setApvPairs( idet->second->getApvNumber()/2 );
	      conn->setFiberLength( static_cast<uint32_t>( idet->second->getFibreLength() ) );
#endif
	    }
	
	    // Add FedChannelConnectionDescription to vector
	    conns.push_back(conn);
	
	  }
	}
      }
      idcu++;
    }
    if ( !found ) { unconnected++; }
  }
  
  edm::LogVerbatim(mlDqmClient_)
    << "[FastFedCablingHistosUsingDb::" << __func__ << "]"
    << " Found " << conns.size() 
    << " connections and "
    << unconnected 
    << " unconnected LLD channels ("
    << 100 * conns.size() / ( conns.size() + unconnected ) 
    << "% of total)";
  
#ifndef USING_NEW_DATABASE_MODEL	
  // Some debug
  std::stringstream ss; 
  ss << "[FastFedCablingHistosUsingDb::" << __func__ << "]"
     << " Dump of " << conns.size() 
     << " FedChannelConnection descriptions: "
     << std::endl;
  SiStripConfigDb::FedConnections::iterator ifed = conns.begin();
  for ( ; ifed != conns.end(); ifed++ ) { (*ifed)->toXML(ss); }
  LogTrace(mlTest_) << ss.str();
#endif
  
}

// -----------------------------------------------------------------------------
/** */
void FastFedCablingHistosUsingDb::update( SiStripConfigDb::FedDescriptions& feds ) {

  // Iterate through feds and disable all channels 
  SiStripConfigDb::FedDescriptions::iterator ifed;
  try {
    for ( ifed = feds.begin(); ifed != feds.end(); ifed++ ) {
      for ( uint16_t ichan = 0; ichan < sistrip::FEDCH_PER_FED; ichan++ ) {
	Fed9U::Fed9UAddress addr( ichan );
	Fed9U::Fed9UAddress addr0( ichan, static_cast<Fed9U::u8>(0) );
	Fed9U::Fed9UAddress addr1( ichan, static_cast<Fed9U::u8>(1) );
	(*ifed)->setFedFeUnitDisable( addr, true );
	(*ifed)->setApvDisable( addr0, true );
	(*ifed)->setApvDisable( addr1, true );
      }
    }
  } catch( ICUtils::ICException& e ) {
    edm::LogWarning(mlDqmClient_) << e.what();
  }
  
  // Counters for number of connected / enabled channels
  uint16_t connected = 0;
  std::map< uint16_t, std::vector<uint16_t> > enabled;

  // Iterate through feds and enable connected channels
  for ( ifed = feds.begin(); ifed != feds.end(); ifed++ ) {
    for ( uint16_t ichan = 0; ichan < sistrip::FEDCH_PER_FED; ichan++ ) {
      
      // Retrieve FEC key from FED-FEC map
      SiStripFedKey fed( static_cast<uint16_t>( (*ifed)->getFedId() ),
			 SiStripFedKey::feUnit(ichan),
			 SiStripFedKey::feChan(ichan) );
      uint32_t fed_key = fed.key();
      
      // Retrieve analysis for given FED id and channel
      map<uint32_t,FastFedCablingAnalysis*>::const_iterator ianal = data_.find(fed_key);
      if ( ianal == data_.end() ) { continue; }
      
      if ( !ianal->second->isValid() ) { continue; }
      
      // Retrieve FED id and channel 
      uint16_t fed_id = ianal->second->fedKey().fedId();
      uint16_t fed_ch = ianal->second->fedKey().fedChannel();
      
      // Enable front-end unit and channel
      Fed9U::Fed9UAddress addr( fed_ch );
      Fed9U::Fed9UAddress addr0( fed_ch, static_cast<Fed9U::u8>(0) );
      Fed9U::Fed9UAddress addr1( fed_ch, static_cast<Fed9U::u8>(1) );
      (*ifed)->setFedFeUnitDisable( addr, false );
      (*ifed)->setApvDisable( addr0, false );
      (*ifed)->setApvDisable( addr1, false );
      connected++;
      enabled[fed_id].push_back(fed_ch);
      
    }
  }
      
  // Some debug 
  edm::LogVerbatim(mlDqmClient_)
    << "[FastFedCablingHistosUsingDb::" << __func__ << "]"
    << " Enabled a total of " << connected 
    << " FED channels and disabled " << feds.size() * 96 - connected 
    << " FED channels (" << 100 * connected / ( feds.size() * 96 )
    << "% of total)";
  
  // Some debug 
  std::stringstream ss;
  ss << "[FastFedCablingHistosUsingDb::" << __func__ << "]"
     << " Dump of enabled FED channels:" 
     << std::endl;
  std::map< uint16_t, std::vector<uint16_t> >::const_iterator fed = enabled.begin();
  for ( ; fed != enabled.end(); fed++ ) {
    ss << " Enabled " << fed->second.size()
       << " channels for FED id " 
       << std::setw(3) << fed->first << ": ";
    if ( !fed->second.empty() ) { 
      uint16_t first = fed->second.front();
      uint16_t last = fed->second.front();
      std::vector<uint16_t>::const_iterator chan = fed->second.begin();
      for ( ; chan != fed->second.end(); chan++ ) { 
	if ( chan != fed->second.begin() ) {
	  if ( *chan != last+1 ) { 
	    ss << std::setw(2) << first << "->" << std::setw(2) << last << ", ";
	    if ( chan != fed->second.end() ) { first = *(chan+1); }
	  } 
	}
	last = *chan;
      }
      if ( first != last ) { ss << std::setw(2) << first << "->" << std::setw(2) << last; }
      ss << std::endl;
    }
  }
  LogTrace(mlDqmClient_) << ss.str();
  
}
