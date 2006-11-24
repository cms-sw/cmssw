// Last commit: $Id: FedConnections.cc,v 1.6 2006/11/03 11:17:24 bainbrid Exp $
// Latest tag:  $Name:  $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripConfigDb/src/FedConnections.cc,v $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <ostream>

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
  stringstream ss; 
  ss << "[SiStripConfigDb::" << __func__ << "]";
  if ( connections_.empty() ) { ss << " Found no FED connections"; }
  else { ss << " Found " << connections_.size() << " FED connections"; }
  if ( !dbParams_.usingDb_ ) { ss << " in " << dbParams_.inputModuleXml_ << " 'module.xml' file"; }
  else { ss << " in database partition '" << dbParams_.partition_ << "'"; }
  if ( connections_.empty() ) { edm::LogWarning(mlConfigDb_) << ss; }
  else { LogTrace(mlConfigDb_) << ss; }
  
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
  
  if ( dbParams_.usingDb_ ) {
    
    try { 
      
      SiStripConfigDb::FedConnections::iterator ifed = connections_.begin();
      for ( ; ifed != connections_.end(); ifed++ ) {
	deviceFactory(__func__)->addFedChannelConnection( *ifed );
      }
      deviceFactory(__func__)->upload();
      
    }
    catch (...) {
      handleException( __func__ );
    }
    
  } else {
    
    ofstream out( dbParams_.outputModuleXml_.c_str() );
    out << "<?xml version=\"1.0\" encoding=\"ISO-8859-1\" ?>" << endl
	<< "<!DOCTYPE TrackerDescription SYSTEM \"http://cmsdoc.cern.ch/cms/cmt/System_aspects/Daq/dtd/trackerdescription.dtd\">" << endl
	<< "<TrackerDescription>" << endl
	<< "<FedChannelList>" << endl;
    SiStripConfigDb::FedConnections::iterator ifed = connections_.begin();
    for ( ; ifed != connections_.end(); ifed++ ) { (*ifed)->toXML(out); out << endl; }
    out << "</FedChannelList>" << endl
	<< "</TrackerDescription>" << endl;
    out.close();

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
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " No FED connections created!";
  }
  
  return static_fed_connections;

}

