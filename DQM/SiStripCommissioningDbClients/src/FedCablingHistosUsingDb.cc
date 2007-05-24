// Last commit: $Id: FedCablingHistosUsingDb.cc,v 1.3 2007/04/04 07:21:08 bainbrid Exp $

#include "DQM/SiStripCommissioningDbClients/interface/FedCablingHistosUsingDb.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
FedCablingHistosUsingDb::FedCablingHistosUsingDb( MonitorUserInterface* mui,
						  const DbParams& params )
  : FedCablingHistograms( mui ),
    CommissioningHistosUsingDb( params )
{
  LogTrace(mlDqmClient_)
    << "[FedCablingHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
FedCablingHistosUsingDb::FedCablingHistosUsingDb( MonitorUserInterface* mui,
						  SiStripConfigDb* const db ) 
  : FedCablingHistograms( mui ),
    CommissioningHistosUsingDb( db )
{
  LogTrace(mlDqmClient_)
    << "[FedCablingHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
FedCablingHistosUsingDb::FedCablingHistosUsingDb( DaqMonitorBEInterface* bei,
						  SiStripConfigDb* const db ) 
  : FedCablingHistograms( bei ),
    CommissioningHistosUsingDb( db )
{
  LogTrace(mlDqmClient_) 
    << "[FedCablingHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
FedCablingHistosUsingDb::~FedCablingHistosUsingDb() {
  LogTrace(mlDqmClient_)
    << "[FedCablingHistosUsingDb::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
void FedCablingHistosUsingDb::uploadToConfigDb() {
  
  if ( !db_ ) {
    edm::LogWarning(mlDqmClient_) 
      << "[FedCablingHistosUsingDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb interface!"
      << " Aborting upload...";
    return;
  }

  // Retrieve descriptions for DCU id and DetId 
  SiStripConfigDb::DeviceDescriptions dcus; 
  db_->getDeviceDescriptions( dcus, DCU ); 
  SiStripConfigDb::DcuDetIdMap detids;
  detids = db_->getDcuDetIdMap(); 
  
  // Update FED connection descriptions
  db_->resetFedConnections();
  const SiStripConfigDb::FedConnections& conns = db_->getFedConnections(); 
  update( const_cast<SiStripConfigDb::FedConnections&>(conns), dcus, detids );
  if ( !test_ ) { 
    edm::LogVerbatim(mlDqmClient_) 
      << "[FedCablingHistosUsingDb::" << __func__ << "]"
      << " Uploading FED connections to DB...";
    db_->uploadFedConnections(true); 
    edm::LogVerbatim(mlDqmClient_) 
      << "[FedCablingHistosUsingDb::" << __func__ << "]"
      << " Completed database upload of " << conns.size() 
      << " FedChannelConnectionDescriptions!";
  } else {
    edm::LogWarning(mlDqmClient_) 
      << "[FedCablingHistosUsingDb::" << __func__ << "]"
      << " TEST only! No FED connections will be uploaded to DB...";
  }
  
  // Update FED descriptions with enabled/disabled channels
  db_->resetFedDescriptions();
  const SiStripConfigDb::FedDescriptions& feds = db_->getFedDescriptions(); 
  update( const_cast<SiStripConfigDb::FedDescriptions&>(feds) );
  if ( !test_ ) { 
    edm::LogVerbatim(mlDqmClient_) 
      << "[FedCablingHistosUsingDb::" << __func__ << "]"
      << " Uploading FED descriptions to DB...";
    db_->uploadFedDescriptions(false); 
    edm::LogVerbatim(mlDqmClient_) 
      << "[FedCablingHistosUsingDb::" << __func__ << "]"
      << " Completed database upload of " << feds.size()
      << " Fed9UDescriptions (with connected channels enabled)!";
  } else {
    edm::LogWarning(mlDqmClient_) 
      << "[FedCablingHistosUsingDb::" << __func__ << "]"
      << " TEST only! No FED descriptions will be uploaded to DB...";
  }
  
}

// -----------------------------------------------------------------------------
/** */
void FedCablingHistosUsingDb::update( SiStripConfigDb::FedConnections& conns,
				      const SiStripConfigDb::DeviceDescriptions& dcus, 
				      const SiStripConfigDb::DcuDetIdMap& detids ) {

  // Check no connections already exist in database
  if ( !conns.empty() ) {
    edm::LogWarning(mlDqmClient_)
      << "[FedCablingHistosUsingDb::" << __func__ << "]"
      << " Found existing FED channel connections!"
      << " No upload to DB will be performed!";
    test_ = true; // Inhibit DB upload
    conns.clear();
  }
    
  // Retrieve and clear FED-FEC mapping in base class
  FedToFecMap& fed_map = const_cast<FedToFecMap&>( mapping() );
  fed_map.clear();

  // Counter for unconnected channels
  uint16_t unconnected = 0;

  // Update FED-FEC mapping in base class, based on analysis results
  map<uint32_t,FedCablingAnalysis*>::const_iterator ianal;
  for ( ianal = data_.begin(); ianal != data_.end(); ianal++ ) {
    
    // Generate FEC key
    SiStripFecKey fec_path( ianal->first );

    // Generate FED key
    uint16_t fed_ch = ianal->second->fedCh();
    SiStripFedKey fed_path( ianal->second->fedId(), 
			    SiStripFedKey::feUnit(fed_ch),
			    SiStripFedKey::feChan(fed_ch) );
    uint32_t fed_key = fed_path.key();
    
    // Check if FedKey is valid 
    if ( fed_path.fedId() == sistrip::invalid_ || 
	 fed_path.fedChannel() == sistrip::invalid_ ) {
      unconnected++;
      continue;
    }
    
    // Add entry to FED-FEC mapping object if FedKey
    if ( fed_map.find(fed_key) != fed_map.end() ) {
      edm::LogWarning(mlDqmClient_)
	<< "[FedCablingHistosUsingDb::" << __func__ << "]"
	<< " FED key 0x" << std::hex 
	<< std::setw(8) << std::setfill('0') << fed_key << std::dec
	<< " with FedId/Ch: " << fed_path.fedId() << "/" 
	<< fed_path.fedChannel() 
	<< " already found! Overwriting...";
    }
    fed_map[fed_key] = ianal->first;
    
    // Create connection object and set member data 
    FedChannelConnectionDescription* conn = new FedChannelConnectionDescription(); 
    conn->setFedId( ianal->second->fedId() );
    conn->setFedChannel( ianal->second->fedCh() );
    conn->setFecSupervisor("");
    conn->setFecSupervisorIP("");
    conn->setFecInstance( fec_path.fecCrate() );
    conn->setSlot( fec_path.fecSlot() );
    conn->setRing( fec_path.fecRing() );
    conn->setCcu( fec_path.ccuAddr() );
    conn->setI2c( fec_path.ccuChan() );
    conn->setApv( SiStripFecKey::i2cAddr(fec_path.channel(),true) );
    conn->setDcuHardId( sistrip::invalid_ );
    conn->setDetId( sistrip::invalid_ );
    conn->setFiberLength( sistrip::invalid_ );
    conn->setApvPairs( sistrip::invalid_ );
    
    // Retrieve DCU id from DB and set member data 
    SiStripConfigDb::DeviceDescriptions::const_iterator idcu;
    for ( idcu = dcus.begin(); idcu != dcus.end(); idcu++ ) {

      // Check if DCU on Front-End Hybrid
      dcuDescription* dcu = dynamic_cast<dcuDescription*>( *idcu );
      if ( !dcu ) { continue; }
      if ( dcu->getDcuType() != "FEH" ) { continue; }

      // Set DCU id if I2C address is consistent
      const SiStripConfigDb::DeviceAddress& addr = db_->deviceAddress(*dcu);
      SiStripFecKey path( addr.fecCrate_ + sistrip::FEC_CRATE_OFFSET, 
			  addr.fecSlot_, 
			  addr.fecRing_ + sistrip::FEC_RING_OFFSET, 
			  addr.ccuAddr_, 
			  addr.ccuChan_ ); 
      if ( path.fecCrate() == fec_path.fecCrate() && 
	   path.fecRing() == fec_path.fecRing() && 
	   path.fecSlot() == fec_path.fecSlot() && 
	   path.ccuAddr() == fec_path.ccuAddr() && 
	   path.ccuChan() == fec_path.ccuChan() ) {
	conn->setDcuHardId( dcu->getDcuHardId() ); 
      }

    }
    
    // Retrieve DetId from DB and set member data 
    SiStripConfigDb::DcuDetIdMap::const_iterator idet = detids.find( conn->getDcuHardId() );
    if ( idet != detids.end() ) { 
      conn->setDetId( idet->second->getDetId() );
      conn->setApvPairs( idet->second->getApvNumber()/2 );
      conn->setFiberLength( static_cast<uint32_t>( idet->second->getFibreLength() ) );
    }
      
    // Add FedChannelConnectionDescription to vector
    conns.push_back(conn);
    
  }

  edm::LogVerbatim(mlDqmClient_)
    << "[FedCablingHistosUsingDb::" << __func__ << "]"
    << " Added " << mapping().size()
    << " entries to FED-FEC mapping object!";
  
  edm::LogVerbatim(mlDqmClient_)
    << "[FedCablingHistosUsingDb::" << __func__ << "]"
    << " Found " << conns.size() 
    << " connections and "
    << unconnected 
    << " unconnected LLD channels ("
    << 100 * conns.size() / ( conns.size() + unconnected ) 
    << "% of total)";
  
  // Some debug
  std::stringstream ss; 
  ss << "[FedCablingHistosUsingDb::" << __func__ << "]"
     << " Dump of " << conns.size() 
     << " FedChannelConnection descriptions: "
     << std::endl;
  SiStripConfigDb::FedConnections::iterator ifed = conns.begin();
  for ( ; ifed != conns.end(); ifed++ ) { (*ifed)->toXML(ss); }
  LogTrace(mlTest_) << ss.str();
  
}

// -----------------------------------------------------------------------------
/** */
void FedCablingHistosUsingDb::update( SiStripConfigDb::FedDescriptions& feds ) {

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
      uint32_t fec_key = 0;
      uint16_t fed_id = static_cast<uint16_t>( (*ifed)->getFedId() );
      uint32_t fed_key = SiStripFedKey( fed_id,
					SiStripFedKey::feUnit(ichan),
					SiStripFedKey::feChan(ichan) ).key();
      FedToFecMap::const_iterator ifec = mapping().find(fed_key);
      if ( ifec != mapping().end() ) { fec_key = ifec->second; }
      else { continue; }
      
      // Enable front-end unit and channel
      map<uint32_t,FedCablingAnalysis*>::const_iterator iter = data_.find( fec_key );
      if ( iter != data_.end() ) { 
	Fed9U::Fed9UAddress addr( ichan );
	Fed9U::Fed9UAddress addr0( ichan, static_cast<Fed9U::u8>(0) );
	Fed9U::Fed9UAddress addr1( ichan, static_cast<Fed9U::u8>(1) );
	(*ifed)->setFedFeUnitDisable( addr, false );
	(*ifed)->setApvDisable( addr0, false );
	(*ifed)->setApvDisable( addr1, false );
	connected++;
	enabled[fed_id].push_back(ichan);
      }
      
    } 
  } 
  
  // Some debug 
  edm::LogVerbatim(mlDqmClient_)
    << "[FedCablingHistosUsingDb::" << __func__ << "]"
    << " Enabled a total of " << connected 
    << " FED channels and disabled " << feds.size() * 96 - connected 
    << " FED channels (" << 100 * connected / ( feds.size() * 96 )
    << "% of total)";
  
  // Some debug 
  std::stringstream ss;
  ss << "[FedCablingHistosUsingDb::" << __func__ << "]"
     << " Dump of enabled FED channels:" 
     << std::endl;
  std::map< uint16_t, std::vector<uint16_t> >::const_iterator fed = enabled.begin();
  for ( ; fed != enabled.end(); fed++ ) {
    ss << " Enabled " << fed->second.size()
       << " channels for FED id " << fed->first << ": ";
    std::vector<uint16_t>::const_iterator chan = fed->second.begin();
    for ( ; chan != fed->second.end(); chan++ ) { ss << *chan << " "; }
    ss << std::endl;
  }
  LogTrace(mlDqmClient_) << ss.str();
  
}
