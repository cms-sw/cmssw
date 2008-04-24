// Last commit: $Id: FedConnections.cc,v 1.21 2008/04/23 12:17:48 bainbrid Exp $

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
  
  try {
    
    if ( !dbParams_.usingDbCache_ ) { 
      
      SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partitions_.begin();
      SiStripDbParams::SiStripPartitions::const_iterator jter = dbParams_.partitions_.end();
      for ( ; iter != jter; ++iter ) {
	
	if ( partition == "" || partition == iter->second.partitionName_ ) {
	  
	  FedConnections::range range = connections_.find( iter->second.partitionName_ );
	  if ( range == connections_.emptyRange() ) {
	    
#ifdef USING_NEW_DATABASE_MODEL
	    
	    // Retrieve connections
	    std::vector<FedConnection*> tmp1;
	    deviceFactory(__func__)->getConnectionDescriptions( iter->second.partitionName_, 
								tmp1,
								iter->second.cabVersion_.first,
								iter->second.cabVersion_.second,
								false ); //@@ do not get DISABLED connections
	    
	    // Make local copy 
	    std::vector<FedConnection*> tmp2;
	    ConnectionFactory::vectorCopyI( tmp2, tmp1, true );
	    
	    // Add to cache
	    connections_.loadNext( iter->second.partitionName_, tmp2 );
	    
#else // USING_NEW_DATABASE_MODEL
	    
	    try { 
	      static bool once = true;
	      if ( once ) { deviceFactory(__func__)->createInputDBAccess(); once = false; }
	    } catch (...) { 
	      handleException( __func__, "Attempted to 'createInputDBAccess' for FED-FEC connections!" );
	    }
	    
	    try {
	      deviceFactory(__func__)->setInputDBVersion( iter->second.partitionName_,
							  iter->second.cabVersion_.first,
							  iter->second.cabVersion_.second );
	    } catch (...) { 
	      std::stringstream ss;
	      ss << "Attempted to 'setInputDBVersion' for partition: " << iter->second.partitionName_;
	      handleException( __func__, ss.str() ); 
	    }
	    
	    std::vector<FedConnection*> tmp;
	    for ( uint16_t iconn = 0; iconn < deviceFactory(__func__)->getNumberOfFedChannel(); iconn++ ) {
	      tmp.push_back( deviceFactory(__func__)->getFedChannelConnection( iconn ) ); 
	    }
	    connections_.loadNext( "", tmp );
	    
#endif // USING_NEW_DATABASE_MODEL

	    // Some debug
	    FedConnections::range conns = connections_.find( iter->second.partitionName_ );
	    std::stringstream ss;
	    ss << "[SiStripConfigDb::" << __func__ << "]"
	       << " Dowloaded " << conns.size() 
	       << " FED connections to local cache for partition \""
	       << iter->second.partitionName_ << "\"."
	       << " (Cache holds connections for " 
	       << connections_.size() << " partitions.)";
	    LogTrace(mlConfigDb_) << ss.str();
	    
	  }
	  
	}

      }

    } else { // Use database cache
	
#ifdef USING_NEW_DATABASE_MODEL
	
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
    
#endif // USING_NEW_DATABASE_MODEL
      
    }
    
  } catch (...) { handleException( __func__ ); }
  
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
     << " Found " << nc << " FED connections";
  if ( !dbParams_.usingDb_ ) { ss << " in " << dbParams_.inputModuleXmlFiles().size() << " 'module.xml' file(s)"; }
  else { if ( !dbParams_.usingDbCache_ )  { ss << " in " << np << " database partition(s)"; } 
  else { ss << " from shared memory name '" << dbParams_.sharedMemory_ << "'"; } }
  if ( connections_.empty() ) { edm::LogWarning(mlConfigDb_) << ss.str(); }
  else { LogTrace(mlConfigDb_) << ss.str(); }

  return conns;
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::addFedConnections( std::string partition, std::vector<FedConnection*>& conns ) {

  if ( !deviceFactory(__func__) ) { return; }

  if ( partition.empty() ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Partition string is empty,"
       << " therefore cannot add FED connections to local cache!"; 
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
  if ( conns.empty() ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Vector of FED connections is empty,"
       << " therefore cannot add FED connections to local cache!"; 
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }

  SiStripDbParams::SiStripPartitions::iterator iter = dbParams_.partitions_.begin();
  SiStripDbParams::SiStripPartitions::iterator jter = dbParams_.partitions_.end();
  for ( ; iter != jter; ++iter ) { if ( partition == iter->second.partitionName_ ) { break; } }
  if ( iter == dbParams_.partitions_.end() ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Partition \"" << partition
       << "\" not found in partition list, "
       << " therefore cannot add FED connections!";
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
  FedConnections::range range = connections_.find( partition );
  if ( range == connections_.emptyRange() ) {
    
    // Make local copy 
    std::vector<FedConnection*> tmp;
#ifdef USING_NEW_DATABASE_MODEL
    ConnectionFactory::vectorCopyI( tmp, conns, true );
#else
    tmp = conns;
#endif
    
    // Add to local cache
    connections_.loadNext( partition, tmp );

    // Some debug
    std::stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " Added " << conns.size() 
       << " FED connections to local cache for partition \""
       << partition << "\"."
       << " (Cache holds FED connections for " 
       << connections_.size() << " partitions.)";
    LogTrace(mlConfigDb_) << ss.str();
    
  } else {
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Partition \"" << partition
       << "\" already found in local cache, "
       << " therefore cannot add new FED connections!";
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
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
      
      SiStripDbParams::SiStripPartitions::iterator iter = dbParams_.partitions_.begin();
      SiStripDbParams::SiStripPartitions::iterator jter = dbParams_.partitions_.end();
      for ( ; iter != jter; ++iter ) {

	if ( partition == "" || partition == iter->second.partitionName_ ) {

	  FedConnections::range range = connections_.find( iter->second.partitionName_ );
	  if ( range != connections_.emptyRange() ) {

	    std::vector<FedConnection*> conns( range.begin(), range.end() );
	    
#ifdef USING_NEW_DATABASE_MODEL
	    deviceFactory(__func__)->setConnectionDescriptions( conns,
								iter->second.partitionName_,
								&(iter->second.cabVersion_.first),
								&(iter->second.cabVersion_.second),
								true ); // new major version
#else 
	    std::vector<FedConnection*>::iterator ifed = conns.begin();
	    std::vector<FedConnection*>::iterator jfed = conns.end();
	    for ( ; ifed != jfed; ++ifed ) {
	      deviceFactory(__func__)->addFedChannelConnection( *ifed );
	    }
	    deviceFactory(__func__)->upload();
#endif

	    // Some debug
	    std::stringstream ss;
	    ss << "[SiStripConfigDb::" << __func__ << "]"
	       << " Uploaded " << conns.size() 
	       << " FED connections to DB/xml for partition \""
	       << iter->second.partitionName_ << "\".";
	    LogTrace(mlConfigDb_) << ss.str();

	  } else {
	    stringstream ss; 
	    ss << "[SiStripConfigDb::" << __func__ << "]" 
	       << " Vector of FED connections is empty for partition \"" 
	       << iter->second.partitionName_
	       << "\", therefore aborting upload for this partition!";
	    edm::LogWarning(mlConfigDb_) << ss.str(); 
	    continue; 
	  }

	} else {
	  // 	  stringstream ss; 
	  // 	  ss << "[SiStripConfigDb::" << __func__ << "]" 
	  // 	     << " Cannot find partition \"" << partition
	  // 	     << "\" in cached partitions list: \""
	  // 	     << dbParams_.partitions( dbParams_.partitions() ) 
	  // 	     << "\", therefore aborting upload for this partition!";
	  // 	  edm::LogWarning(mlConfigDb_) << ss.str(); 
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
    SiStripDbParams::SiStripPartitions::iterator iter = dbParams_.partitions_.begin();
    SiStripDbParams::SiStripPartitions::iterator jter = dbParams_.partitions_.end();
    for ( ; iter != jter; ++iter ) {
      if ( partition == "" || partition == iter->second.partitionName_ ) {
	FedConnections::range range = connections_.find( iter->second.partitionName_ );
	if ( range != connections_.emptyRange() ) {
	  std::vector<FedConnection*>::const_iterator ifed = range.begin();
	  std::vector<FedConnection*>::const_iterator jfed = range.end();
	  for ( ; ifed != jfed; ++ifed ) { (*ifed)->toXML(out); out << endl; }
	}
      }
    }
    out << "</FedChannelList>" << endl
	<< "</TrackerDescription>" << endl;
    out.close();

#endif

  }

  allowCalibUpload_ = true;
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::clearFedConnections( std::string partition ) {
  LogTrace(mlConfigDb_) << "[SiStripConfigDb::" << __func__ << "]";
  
  if ( connections_.empty() ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Found no cached FED connections!"; 
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
  // Reproduce temporary cache for "all partitions except specified one" (or clear all if none specified)
  FedConnections temporary_cache;
  if ( partition == ""  ) { temporary_cache = FedConnections(); }
  else {
    SiStripDbParams::SiStripPartitions::iterator iter = dbParams_.partitions_.begin();
    SiStripDbParams::SiStripPartitions::iterator jter = dbParams_.partitions_.end();
    for ( ; iter != jter; ++iter ) {
      if ( partition != iter->second.partitionName_ ) {
	FedConnections::range range = connections_.find( iter->second.partitionName_ );
	if ( range != connections_.emptyRange() ) {
	  temporary_cache.loadNext( partition, std::vector<FedConnection*>( range.begin(), range.end() ) );
	} else {
	  // 	  stringstream ss; 
	  // 	  ss << "[SiStripConfigDb::" << __func__ << "]" 
	  // 	     << " Cannot find partition \"" << iter->second.partitionName_
	  // 	     << "\" in local cache!";
	  // 	  edm::LogWarning(mlConfigDb_) << ss.str(); 
	}
      }
    }
  }

  // Delete objects in local cache for specified partition (or all if not specified) 
  FedConnections::range conns;
  if ( partition == "" ) { 
    conns = FedConnections::range( connections_.find( dbParams_.partitions_.begin()->second.partitionName_ ).begin(),
				   connections_.find( dbParams_.partitions_.rbegin()->second.partitionName_ ).end() );
  } else {
    SiStripDbParams::SiStripPartitions::iterator iter = dbParams_.partitions_.begin();
    SiStripDbParams::SiStripPartitions::iterator jter = dbParams_.partitions_.end();
    for ( ; iter != jter; ++iter ) { if ( partition == iter->second.partitionName_ ) { break; } }
    conns = connections_.find( iter->second.partitionName_ );
  }
  
  if ( conns != connections_.emptyRange() ) {

#ifdef USING_NEW_DATABASE_MODEL	
    std::vector<FedConnection*>::const_iterator ifed = conns.begin();
    std::vector<FedConnection*>::const_iterator jfed = conns.end();
    for ( ; ifed != jfed; ++ifed ) { if ( *ifed ) { delete *ifed; } }
#endif
    
  } else {
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]";
    if ( partition == "" ) { ss << " Found no FED connections in local cache!"; }
    else { ss << " Found no FED connections in local cache for partition \"" << partition << "\"!"; }
    edm::LogWarning(mlConfigDb_) << ss.str(); 
  }
  
  // Overwrite local cache with temporary cache
  connections_ = temporary_cache; 

}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::printFedConnections( std::string partition ) {

  std::stringstream ss;
  ss << "[SiStripConfigDb::" << __func__ << "]"
     << " Contents of FedConnections container:" << std::endl;
  ss << " Number of partitions: " << connections_.size() << std::endl;

  // Loop through partitions
  uint16_t cntr = 0;
  FedConnections::const_iterator iconn = connections_.begin();
  FedConnections::const_iterator jconn = connections_.end();
  for ( ; iconn != jconn; ++iconn ) {

    cntr++;
    if ( partition == "" || partition == iconn->first ) {
      
      ss << "  Partition number   : " << cntr << " (out of " << connections_.size() << ")" << std::endl;
      ss << "  Partition name     : " << iconn->first << std::endl;
      ss << "  Num of connections : " << iconn->second.size() << std::endl;

      // Extract FED ids and channels
      std::map< uint16_t, vector<uint16_t> > feds;
      std::vector<FedConnection*>::const_iterator iter = iconn->second.begin();
      std::vector<FedConnection*>::const_iterator jter = iconn->second.end();
      for ( ; iter != jter; ++iter ) { 
	if ( *iter ) { 
	  uint16_t fed_id = (*iter)->getFedId();
	  uint16_t fed_ch = (*iter)->getFedChannel();
	  if ( find( feds[fed_id].begin(), feds[fed_id].end(), fed_ch ) == feds[fed_id].end() ) { 
	    feds[fed_id].push_back( fed_ch ); 
	  }
	}
      }
      
      // Sort contents
      std::map< uint16_t, std::vector<uint16_t> > tmp;
      std::map< uint16_t, std::vector<uint16_t> >::const_iterator ii = feds.begin();
      std::map< uint16_t, std::vector<uint16_t> >::const_iterator jj = feds.end();
      for ( ; ii != jj; ++ii ) {
	std::vector<uint16_t> temp = ii->second;
	std::sort( temp.begin(), temp.end() );
	std::vector<uint16_t>::const_iterator iii = temp.begin();
	std::vector<uint16_t>::const_iterator jjj = temp.end();
	for ( ; iii != jjj; ++iii ) { tmp[ii->first].push_back( *iii ); }
      }
      feds.clear();
      feds = tmp;
      
      // Print FED ids and channels
      std::map< uint16_t, std::vector<uint16_t> >::const_iterator ifed = feds.begin();
      std::map< uint16_t, std::vector<uint16_t> >::const_iterator jfed = feds.end();
      for ( ; ifed != jfed; ++ifed ) {
	ss << "  Found " << std::setw(2) << ifed->second.size()
	   << " channels for FED id " << std::setw(3) << ifed->first << " : ";
	if ( !ifed->second.empty() ) { 
	  uint16_t first = ifed->second.front();
	  uint16_t last = ifed->second.front();
	  std::vector<uint16_t>::const_iterator ichan = ifed->second.begin();
	  std::vector<uint16_t>::const_iterator jchan = ifed->second.end();
	  for ( ; ichan != jchan; ++ichan ) { 
	    if ( ichan != ifed->second.begin() ) {
	      if ( *ichan != last+1 ) { 
		ss << std::setw(2) << first << "->" << std::setw(2) << last << ", ";
		if ( ichan != ifed->second.end() ) { first = *(ichan+1); }
	      } 
	    }
	    last = *ichan;
	  }
	  if ( first != last ) { ss << std::setw(2) << first << "->" << std::setw(2) << last; }
	  ss << std::endl;
	}
      }
      
    }
    
  }
  
  LogTrace(mlConfigDb_) << ss.str();

}
