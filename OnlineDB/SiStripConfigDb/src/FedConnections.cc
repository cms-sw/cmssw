// Last commit: $Id: FedConnections.cc,v 1.1 2006/06/30 06:57:52 bainbrid Exp $
// Latest tag:  $Name: V00-01-01 $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripConfigDb/src/FedConnections.cc,v $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"

using namespace std;

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::FedConnections& SiStripConfigDb::getFedConnections() {
  string method = "SiStripConfigDb::getFedConnections";

  // If reset flag set, return contents of local cache
  if ( !resetConnections_ ) { return connections_; }
  
  if ( usingDb_ ) { 
    try {
      deviceFactory(method)->setInputDBVersion( partition_.name_,
						partition_.major_,
						partition_.minor_ );
    }
    catch (...) { 
      string info = "Problems setting input DB version!";
      handleException( method, info ); 
    }
  }
  
  try {
    for ( int iconn = 0; iconn < deviceFactory(method)->getNumberOfFedChannel(); iconn++ ) {
      connections_.push_back( deviceFactory(method)->getFedChannelConnection( iconn ) ); 
    }
    resetConnections_ = false;
  }
  catch (...) { 
    string info = "Problems retrieving connection descriptions!";
    handleException( method, info ); 
  }
  
  stringstream ss; 
  if ( connections_.empty() ) {
    ss << "["<<method<<"] No FED connections found";
    if ( !usingDb_ ) { ss << " in input 'module.xml' file " << inputModuleXml_; }
    else { ss << " in database partition '" << partition_.name_ << "'"; }
    edm::LogError(logCategory_) << ss.str();
    throw cms::Exception(logCategory_) << ss.str();
  } else {
    ss << "["<<method<<"]"
       << " Found " << connections_.size() << " FED connections";
    if ( !usingDb_ ) { ss << " in input 'module.xml' file " << inputModuleXml_; }
    else { ss << " in database partition '" << partition_.name_ << "'"; }
    edm::LogInfo(logCategory_) << ss.str();
  }
  
  return connections_;
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::resetFedConnections() {
  //string method = "SiStripConfigDb::resetFedConnections";
  //if ( !deviceFactory(method) ) { return; }
  //deviceFactory(method)->getTrackerParser()->purge(); 
  connections_.clear(); 
  resetConnections_ = true;
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::uploadFedConnections( bool new_major_version ) {
  string method = "SiStripConfigDb::uploadFedConnections";

  if ( !deviceFactory(method) ) { return; }

  try { 

    if ( usingDb_ ) {
      deviceFactory(method)->setInputDBVersion( partition_.name_, //@@ ???
					  partition_.major_,
					  partition_.minor_ );
    }

    SiStripConfigDb::FedConnections::iterator ifed = connections_.begin();
    for ( ; ifed != connections_.end(); ifed++ ) {
      deviceFactory(method)->addFedChannelConnection( *ifed );
    }
    deviceFactory(method)->upload();

  }
  catch (...) {
    handleException( method );
  }

}

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::FedConnections& SiStripConfigDb::createFedConnections( const SiStripFecCabling& fec_cabling ) {
  string method = "SiStripConfigDb::createFedConnections";
  
  // Static container
  static FedConnections static_fed_connections;
  static_fed_connections.clear();
  
  // Create FED cabling from FEC cabling
  vector<FedChannelConnection> connections;
  fec_cabling.connections( connections );
  SiStripFedCabling fed_cabling( connections );

  // Iterate through connection descriptions
  vector<FedChannelConnection>::iterator iconn = connections.begin();
  for ( ; iconn != connections.end(); iconn++ ) { 
    FedChannelConnectionDescription* desc = new FedChannelConnectionDescription();
    desc->setFedId( iconn->fedId() );
    desc->setFedChannel( iconn->fedCh() );
    desc->setFecSupervisor( "" ); //@@
    desc->setFecSupervisorIP( "" ); //@@
    desc->setFecInstance( iconn->fecCrate() );
    desc->setSlot( iconn->fecSlot() );
    desc->setRing( iconn->fecRing() );
    desc->setCcu( iconn->ccuAddr() );
    desc->setI2c( iconn->ccuChan() );
    desc->setApv( iconn->i2cAddr(0) );
    desc->setDcuHardId( iconn->dcuId() );
    desc->setDetId( iconn->detId() );
    desc->setFiberLength( 0 ); //@@
    desc->setApvPairs( iconn->nApvPairs() );
    static_fed_connections.push_back( desc );
  }
  
  if ( static_fed_connections.empty() ) {
    stringstream ss;
    ss << "["<<method<<"] No FED connections created!";
    edm::LogError(logCategory_) << ss.str() << "\n";
    //throw cms::Exception(logCategory_) << ss.str() << "\n";
  }
  
  return static_fed_connections;

}

