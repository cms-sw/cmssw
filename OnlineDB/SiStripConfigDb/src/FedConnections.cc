// Last commit: $Id: FedConnections.cc,v 1.2 2006/07/26 11:27:19 bainbrid Exp $
// Latest tag:  $Name: V00-01-02 $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripConfigDb/src/FedConnections.cc,v $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"

using namespace std;

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::FedConnections& SiStripConfigDb::getFedConnections() {

  if ( !deviceFactory(__FUNCTION__) ) { return connections_; }
  if ( !resetConnections_ ) { return connections_; }
  
  try {
    for ( int iconn = 0; iconn < deviceFactory(__FUNCTION__)->getNumberOfFedChannel(); iconn++ ) {
      connections_.push_back( deviceFactory(__FUNCTION__)->getFedChannelConnection( iconn ) ); 
    }
    resetConnections_ = false;
  }
  catch (...) { 
    handleException( __FUNCTION__, "Problems retrieving connection descriptions!" ); 
  }
  
  stringstream ss; 
  if ( connections_.empty() ) {
    ss << "[" << __PRETTY_FUNCTION__ << "] No FED connections found";
    if ( !usingDb_ ) { ss << " in input 'module.xml' file " << inputModuleXml_; }
    else { ss << " in database partition '" << partition_.name_ << "'"; }
    edm::LogError(logCategory_) << ss.str();
  } else {
    ss << "[" << __PRETTY_FUNCTION__ << "]"
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
  connections_.clear(); 
  resetConnections_ = true;
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::uploadFedConnections( bool new_major_version ) {

  if ( !deviceFactory(__FUNCTION__) ) { return; }

  try { 

    if ( usingDb_ ) {
      deviceFactory(__FUNCTION__)->setInputDBVersion( partition_.name_, //@@ ???
						      partition_.major_,
						      partition_.minor_ );
    }

    SiStripConfigDb::FedConnections::iterator ifed = connections_.begin();
    for ( ; ifed != connections_.end(); ifed++ ) {
      deviceFactory(__FUNCTION__)->addFedChannelConnection( *ifed );
    }
    deviceFactory(__FUNCTION__)->upload();

  }
  catch (...) {
    handleException( __FUNCTION__ );
  }

}

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::FedConnections& SiStripConfigDb::createFedConnections( const SiStripFecCabling& fec_cabling ) {
  
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
    ss << "[" << __PRETTY_FUNCTION__ << "] No FED connections created!";
    edm::LogError(logCategory_) << ss.str() << "\n";
  }
  
  return static_fed_connections;

}

