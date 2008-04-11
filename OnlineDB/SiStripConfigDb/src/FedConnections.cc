// Last commit: $Id: FedConnections.cc,v 1.15 2008/03/28 15:31:15 bainbrid Exp $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <ostream>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
SiStripConfigDb::FedConnections::range SiStripConfigDb::getFedConnections( std::string partition ) {

  // Check
  if ( ( !dbParams_.usingDbCache_ && !deviceFactory(__func__) ) ||
       (  dbParams_.usingDbCache_ && !databaseCache(__func__) ) ) { 
    return connections_.emptyRange();
  }
  
  if ( connections_.empty() ) {
    
    try {
      
#ifdef USING_NEW_DATABASE_MODEL
      
      if ( !dbParams_.usingDbCache_ ) { 
	
	SiStripDbParams::SiStripPartitions::const_iterator ii = dbParams_.partitions_.begin();
	SiStripDbParams::SiStripPartitions::const_iterator jj = dbParams_.partitions_.end();
	for ( ; ii != jj; ++ii ) {
	  
	  // Retrieve conections
	  std::vector<FedConnection*> tmp1;
	  deviceFactory(__func__)->getConnectionDescriptions( ii->second.partitionName_, 
							      tmp1,
							      ii->second.cabMajor_,
							      ii->second.cabMinor_,
							      false ); //@@ do not get DISABLED connections
	  
	  // Make local copy 
	  std::vector<FedConnection*> tmp2;
	  ConnectionFactory::vectorCopyI( tmp2, tmp1, true );

	  // Add to cache
	  connections_.loadNext( ii->second.partitionName_, tmp2 );
	  
	}

	// Some debug
	{ 
	  std::stringstream ss;
	  ss << "[SiStripConfigDb::" << __func__ << "]"
	     << " Contents of FedConnections container:" << std::endl;
	  ss << " Number of partitions: " << connections_.size() << std::endl;
	  uint16_t cntr = 0;
	  FedConnections::const_iterator ii = connections_.begin();
	  FedConnections::const_iterator jj = connections_.end();
	  for ( ; ii != jj; ++ii ) {
	    ss << "  PartitionNumber : " << cntr << " (out of " << connections_.size() << ")" << std::endl;
	    ss << "  PartitionName   : " << ii->first << std::endl;
	    ss << "  NumOfConnections: " << ii->second.size() << std::endl;
	    std::vector<uint16_t> feds;
 	    std::vector<FedConnection*>::const_iterator iii = ii->second.begin();
 	    std::vector<FedConnection*>::const_iterator jjj = ii->second.end();
 	    for ( ; iii != jjj; ++iii ) { 
	      if ( *iii ) { 
		if ( find( feds.begin(), feds.end(), (*iii)->getFedId() ) == feds.end() ) { 
		  feds.push_back( (*iii)->getFedId() ); 
		}
	      }
	    }
	    ss << "  FedIds: ";
	    std::vector<uint16_t>::const_iterator iiii = feds.begin();
	    std::vector<uint16_t>::const_iterator jjjj = feds.end();
	    for ( ; iiii != jjjj; ++iiii ) {
	      ( iiii == feds.begin() ) ? ( ss << *iiii ) : ( ss << ", " << *iiii); 
	    }
	    ss << std::endl;
	    cntr++;
	  }
	  LogTrace(mlConfigDb_) << ss.str();
	}
	
	
      } else {
	
#ifdef USING_DATABASE_CACHE
	
	std::vector<FedConnection*>* tmp1 = databaseCache(__func__)->getConnections();
	
	if ( tmp1 ) { 
	
	  // Make local copy 
	  std::vector<FedConnection*> tmp2;
	  ConnectionFactory::vectorCopyI( tmp2, *tmp1, true );
	  
	  // Add to cache
	  connections_.loadNext( "", tmp2 );
	  
	} else {
	  edm::LogWarning(mlConfigDb_)
	    << "[SiStripConfigDb::" << __func__ << "]"
	    << " NULL pointer to FedConnections vector!";
	}
	
#endif // USING_DATABASE_CACHE
	
      }
      
#else // not USING_NEW_DATABASE_MODEL
      
      std::vector<FedConnection*> tmp;
      for ( uint16_t iconn = 0; iconn < deviceFactory(__func__)->getNumberOfFedChannel(); iconn++ ) {
	tmp.push_back( deviceFactory(__func__)->getFedChannelConnection( iconn ) ); 
      }
      connections_.loadNext( "", tmp2 );
      
#endif // USING_NEW_DATABASE_MODEL
      
    } catch (...) { handleException( __func__ ); }
    
  }

  // Create range object
  uint16_t np = 0;
  uint16_t nc = 0;
  FedConnections::range conns;
  if ( partition != "" ) { 
    conns = connections_.find( partition );
    np = 1;
    nc = conns.size();
  } else { 
    conns = FedConnections::range( connections_.find( dbParams_.partitions_.begin()->second.partitionName_ ).begin(),
				   connections_.find( dbParams_.partitions_.rbegin()->second.partitionName_ ).end() );
    np = connections_.size();
    nc = conns.size();
  }
  
  stringstream ss; 
  ss << "[SiStripConfigDb::" << __func__ << "]"
     << " Found " << nc << " FED connections " << nc;
  if ( !dbParams_.usingDb_ ) { ss << " in " << dbParams_.inputModuleXmlFiles().size() << " 'module.xml' files"; }
  else { if ( !dbParams_.usingDbCache_ )  { ss << " in " << np << " database partitions"; } 
  else { ss << " from shared memory name '" << dbParams_.sharedMemory_ << "'"; } }
  if ( connections_.empty() ) { edm::LogWarning(mlConfigDb_) << ss.str(); }
  else { LogTrace(mlConfigDb_) << ss.str(); }

  return conns;
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::uploadFedConnections( std::string partition ) {

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
      
      SiStripDbParams::SiStripPartitions::iterator ii = dbParams_.partitions_.begin();
      SiStripDbParams::SiStripPartitions::iterator jj = dbParams_.partitions_.end();
      for ( ; ii != jj; ++ii ) {

	if ( partition == "" || partition == ii->second.partitionName_ ) {
	  
	  FedConnections::range range = connections_.find( ii->second.partitionName_ );
	  if ( range != connections_.emptyRange() ) {
	    std::vector<FedConnection*> conns( range.begin(), range.end() );
	    
#ifdef USING_NEW_DATABASE_MODEL
	    
	    deviceFactory(__func__)->setConnectionDescriptions( conns,
								ii->second.partitionName_,
								&(ii->second.cabMajor_),
								&(ii->second.cabMinor_),
								true ); // new major version
	    
#else 
	    std::vector<FedConnection*>::iterator ifed = conns.begin();
	    std::vector<FedConnection*>::iterator jfed = conns.end();
	    for ( ; ifed != jfed; ++ifed ) {
	      deviceFactory(__func__)->addFedChannelConnection( *ifed );
	    }
	    deviceFactory(__func__)->upload();
#endif
	  } else {
	  }

	} else {
	}

      }

    } catch (...) { handleException( __func__ ); }
    
  } else {
    
#ifndef USING_NEW_DATABASE_MODEL
    ofstream out( dbParams_.outputModuleXml_.c_str() );
    out << "<?xml version=\"1.0\" encoding=\"ISO-8859-1\" ?>" << endl
	<< "<!DOCTYPE TrackerDescription SYSTEM \"http://cmsdoc.cern.ch/cms/cmt/System_aspects/Daq/dtd/trackerdescription.dtd\">" << endl
	<< "<TrackerDescription>" << endl
	<< "<FedChannelList>" << endl;
    FedConnections::iterator ifed = connections_.begin();
    FedConnections::iterator jfed = connections_.end();
    for ( ; ifed != jfed; ++ifed ) { (*ifed)->toXML(out); out << endl; }
    out << "</FedChannelList>" << endl
	<< "</TrackerDescription>" << endl;
    out.close();
#endif

  }

  allowCalibUpload_ = true;
  
}

