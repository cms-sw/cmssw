// Last commit: $Id: FedConnections.cc,v 1.3 2006/08/31 19:49:41 bainbrid Exp $
// Latest tag:  $Name:  $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripConfigDb/src/FedConnections.cc,v $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::FedConnections& SiStripConfigDb::getFedConnections() {

  if ( !deviceFactory(__func__) ) { return connections_; }
  if ( !resetConnections_ ) { return connections_; }
  
  try {
    for ( int iconn = 0; iconn < deviceFactory(__func__)->getNumberOfFedChannel(); iconn++ ) {
      connections_.push_back( deviceFactory(__func__)->getFedChannelConnection( iconn ) ); 
    }
    resetConnections_ = false;
  }
  catch (...) { 
    handleException( __func__, "Problems retrieving connection descriptions!" ); 
  }
  
  // Debug 
  ostringstream os; 
  if ( connections_.empty() ) { os << " Found no FED connections"; }
  else { os << " Found " << connections_.size() << " FED connections"; }
  if ( !usingDb_ ) { os << " in " << inputModuleXml_.size() << " 'module.xml' file"; }
  else { os << " in database partition '" << partition_.name_ << "'"; }
  if ( connections_.empty() ) { edm::LogError(mlConfigDb_) << os; }
  else { LogTrace(mlConfigDb_) << os; }
  
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

  if ( !deviceFactory(__func__) ) { return; }

  try { 

    if ( usingDb_ ) {
      deviceFactory(__func__)->setInputDBVersion( partition_.name_, //@@ ???
						  partition_.major_,
						  partition_.minor_ );
    }

    SiStripConfigDb::FedConnections::iterator ifed = connections_.begin();
    for ( ; ifed != connections_.end(); ifed++ ) {
      deviceFactory(__func__)->addFedChannelConnection( *ifed );
    }
    deviceFactory(__func__)->upload();

  }
  catch (...) {
    handleException( __func__ );
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
    edm::LogError(mlConfigDb_) << "No FED connections created!";
  }
  
  return static_fed_connections;

}

