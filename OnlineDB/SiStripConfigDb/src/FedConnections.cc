// Last commit: $Id: $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <ostream>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::FedConnections& SiStripConfigDb::getFedConnections() {

  connections_.clear();

  if ( ( !dbParams_.usingDbCache_ && !deviceFactory(__func__) ) ||
       (  dbParams_.usingDbCache_ && !databaseCache(__func__) ) ) { return connections_; }
  
  try {

#ifdef USING_NEW_DATABASE_MODEL

    if ( !dbParams_.usingDbCache_ ) { 

      deviceFactory(__func__)->getConnectionDescriptions( dbParams_.partitions_.front(), 
							  connections_,
							  dbParams_.cabMajor_,
							  dbParams_.cabMinor_,
							  false ); //@@ do not get DISABLED connections

    } else {

#ifdef USING_DATABASE_CACHE
      FedConnections* tmp = databaseCache(__func__)->getConnections();
      if ( tmp ) { connections_ = *tmp; }
      else {
	edm::LogWarning(mlConfigDb_)
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " NULL pointer to FedConnections vector!";
      }

#endif

    }
      
#else

    for ( uint16_t iconn = 0; iconn < deviceFactory(__func__)->getNumberOfFedChannel(); iconn++ ) {
      connections_.push_back( deviceFactory(__func__)->getFedChannelConnection( iconn ) ); 
    }

#endif

  } catch (...) { handleException( __func__ ); }
  
  stringstream ss; 
  ss << "[SiStripConfigDb::" << __func__ << "]"
     << " Found " << connections_.size() 
     << " FED connections"; 
  if ( !dbParams_.usingDb_ ) { ss << " in " << dbParams_.inputModuleXml_ << " 'module.xml' file"; }
  else { if ( !dbParams_.usingDbCache_ )  { ss << " in database partition '" << dbParams_.partitions_.front() << "'"; } 
  else { ss << " from shared memory name '" << dbParams_.sharedMemory_ << "'"; } }
  if ( connections_.empty() ) { edm::LogWarning(mlConfigDb_) << ss.str(); }
  else { LogTrace(mlConfigDb_) << ss.str(); }
  
  return connections_;

}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::uploadFedConnections( bool new_major_version ) {

  if ( dbParams_.usingDbCache_ ) {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]" 
      << " Using database cache! No uploads allowed!"; 
    return;
  }

  if ( !deviceFactory(__func__) ) { return; }
  
  if ( connections_.empty() ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Found no cached FED connections, therefore no upload!"; 
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }

  if ( dbParams_.usingDb_ ) {
    
    try { 
      
#ifdef USING_NEW_DATABASE_MODEL
      deviceFactory(__func__)->setConnectionDescriptions( connections_,
							  dbParams_.partitions_.front(), 
							  &(dbParams_.cabMajor_),
							  &(dbParams_.cabMinor_),
							  new_major_version );
#else 
      SiStripConfigDb::FedConnections::iterator ifed = connections_.begin();
      for ( ; ifed != connections_.end(); ifed++ ) {
	deviceFactory(__func__)->addFedChannelConnection( *ifed );
      }
      deviceFactory(__func__)->upload();
#endif
      
    } catch (...) { handleException( __func__ ); }
    
  } else {
    
#ifndef USING_NEW_DATABASE_MODEL
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
#endif

  }

  allowCalibUpload_ = true;
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::createFedConnections( const SiStripFecCabling& fec_cabling ) {
  
  // Clear cache 
  connections_.clear();
  
  // Create FED cabling from FEC cabling
  vector<FedChannelConnection> connections;
  fec_cabling.connections( connections );
  SiStripFedCabling fed_cabling( connections );

  // Iterate through connection descriptions
  vector<FedChannelConnection>::iterator iconn = connections.begin();
  for ( ; iconn != connections.end(); iconn++ ) { 
#ifdef USING_NEW_DATABASE_MODEL
    ConnectionDescription* desc = new ConnectionDescription();
    desc->setFedId( iconn->fedId() );
    desc->setFedChannel( iconn->fedCh() );
    desc->setFecHardwareId( "" ); //@@
    desc->setFecCrateSlot( iconn->fecCrate() );
    desc->setFecSlot( iconn->fecSlot() );
    desc->setRingSlot( iconn->fecRing() );
    desc->setCcuAddress( iconn->ccuAddr() );
    desc->setI2cChannel( iconn->ccuChan() );
    desc->setApvAddress( iconn->i2cAddr(0) );
    desc->setDcuHardId( iconn->dcuId() );
    //@@ desc->setDetId( iconn->detId() );
    //@@ desc->setApvPairs( iconn->nApvPairs() );
    //@@ desc->setFiberLength( 0 ); //@@
#else
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
    desc->setApvPairs( iconn->nApvPairs() );
    desc->setFiberLength( 0 ); //@@
#endif
    connections_.push_back( desc );
  }
  
  if ( connections_.empty() ) {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " No FED connections created!";
  }
  
}

