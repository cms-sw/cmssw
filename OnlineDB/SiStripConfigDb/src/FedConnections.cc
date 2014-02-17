// Last commit: $Id: FedConnections.cc,v 1.35 2011/09/02 11:25:25 eulisse Exp $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <ostream>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
SiStripConfigDb::FedConnectionsRange SiStripConfigDb::getFedConnections( std::string partition ) {

  // Check
  if ( ( !dbParams_.usingDbCache() && !deviceFactory(__func__) ) ||
       (  dbParams_.usingDbCache() && !databaseCache(__func__) ) ) { 
    return connections_.emptyRange();
  }
  
  try {
    
    if ( !dbParams_.usingDbCache() ) { 
      
      SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partitions().begin();
      SiStripDbParams::SiStripPartitions::const_iterator jter = dbParams_.partitions().end();
      for ( ; iter != jter; ++iter ) {
	
	if ( partition == "" || partition == iter->second.partitionName() ) {

	  if ( iter->second.partitionName() == SiStripPartition::defaultPartitionName_ ) { continue; }
	  
	  FedConnectionsRange range = connections_.find( iter->second.partitionName() );
	  if ( range == connections_.emptyRange() ) {

	    FedConnectionsV tmp2;
	    
	    // Retrieve connections
	    FedConnectionsV tmp1;
	    deviceFactory(__func__)->getConnectionDescriptions( iter->second.partitionName(), 
								tmp1,
								iter->second.cabVersion().first,
								iter->second.cabVersion().second,
								//#ifdef USING_DATABASE_MASKING
							        iter->second.maskVersion().first,
							        iter->second.maskVersion().second,
								//#endif
								false ); //@@ do not get DISABLED connections
	    
	    // Make local copy 
	    ConnectionFactory::vectorCopyI( tmp2, tmp1, true );
	    
	    // Add to cache
	    connections_.loadNext( iter->second.partitionName(), tmp2 );
	    
	    // Some debug
	    FedConnectionsRange conns = connections_.find( iter->second.partitionName() );
	    std::stringstream ss;
	    ss << "[SiStripConfigDb::" << __func__ << "]"
	       << " Downloaded " << conns.size() 
	       << " FED connections to local cache for partition \""
	       << iter->second.partitionName() << "\"" << std::endl;
	    ss << "[SiStripConfigDb::" << __func__ << "]"
	       << " Cache holds FED connections for " 
	       << connections_.size() << " partitions.";
	    LogTrace(mlConfigDb_) << ss.str();
	    
	  }
	  
	}

      }

    } else { // Use database cache
	
      FedConnectionsV* tmp1 = databaseCache(__func__)->getConnections();
      
      if ( tmp1 ) { 
	
	// Make local copy 
	FedConnectionsV tmp2;
	ConnectionFactory::vectorCopyI( tmp2, *tmp1, true );
	
	// Add to cache
	connections_.loadNext( SiStripPartition::defaultPartitionName_, tmp2 );
	
      } else {
	edm::LogWarning(mlConfigDb_)
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " NULL pointer to FedConnections vector!";
      }
      
    }
    
  } catch (...) { handleException( __func__ ); }
  
  // Create range object
  uint16_t np = 0;
  uint16_t nc = 0;
  FedConnectionsRange conns;
  if ( partition != "" ) { 
    conns = connections_.find( partition );
    np = 1;
    nc = conns.size();
  } else { 
    if ( !connections_.empty() ) {
      conns = FedConnectionsRange( connections_.find( dbParams_.partitions().begin()->second.partitionName() ).begin(),
				   connections_.find( (--(dbParams_.partitions().end()))->second.partitionName() ).end() );
    } else { conns = connections_.emptyRange(); }
    np = connections_.size();
    nc = conns.size();
  }
  
  stringstream ss; 
  ss << "[SiStripConfigDb::" << __func__ << "]"
     << " Found " << nc << " FED connections";
  if ( !dbParams_.usingDb() ) { ss << " in " << dbParams_.inputModuleXmlFiles().size() << " 'module.xml' file(s)"; }
  else { if ( !dbParams_.usingDbCache() )  { ss << " in " << np << " database partition(s)"; } 
  else { ss << " from shared memory name '" << dbParams_.sharedMemory() << "'"; } }
  if ( connections_.empty() ) { edm::LogWarning(mlConfigDb_) << ss.str(); }
  else { LogTrace(mlConfigDb_) << ss.str(); }

  return conns;
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::addFedConnections( std::string partition, FedConnectionsV& conns ) {

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

  SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partitions().begin();
  SiStripDbParams::SiStripPartitions::const_iterator jter = dbParams_.partitions().end();
  for ( ; iter != jter; ++iter ) { if ( partition == iter->second.partitionName() ) { break; } }
  if ( iter == dbParams_.partitions().end() ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Partition \"" << partition
       << "\" not found in partition list, "
       << " therefore cannot add FED connections!";
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
  FedConnectionsRange range = connections_.find( partition );
  if ( range == connections_.emptyRange() ) {
    
    // Make local copy 
    FedConnectionsV tmp;
    ConnectionFactory::vectorCopyI( tmp, conns, true );
    
    // Add to local cache
    connections_.loadNext( partition, tmp );

    // Some debug
    std::stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " Added " << conns.size() 
       << " FED connections to local cache for partition \""
       << partition << "\"" << std::endl;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " Cache holds FED connections for " 
       << connections_.size() << " partitions.";
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

  if ( dbParams_.usingDbCache() ) {
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

  if ( dbParams_.usingDb() ) {
    
    try { 
      
      SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partitions().begin();
      SiStripDbParams::SiStripPartitions::const_iterator jter = dbParams_.partitions().end();
      for ( ; iter != jter; ++iter ) {

	if ( partition == "" || partition == iter->second.partitionName() ) {

	  FedConnectionsRange range = connections_.find( iter->second.partitionName() );
	  if ( range != connections_.emptyRange() ) {

	    FedConnectionsV conns( range.begin(), range.end() );
	    
            SiStripPartition::Versions cabVersion = iter->second.cabVersion();
	    deviceFactory(__func__)->setConnectionDescriptions( conns,
								iter->second.partitionName(),
								&(cabVersion.first),
								&(cabVersion.second),
								true ); // new major version

	    // Some debug
	    std::stringstream ss;
	    ss << "[SiStripConfigDb::" << __func__ << "]"
	       << " Uploaded " << conns.size() 
	       << " FED connections to database for partition \""
	       << iter->second.partitionName() << "\".";
	    LogTrace(mlConfigDb_) << ss.str();

	  } else {
	    stringstream ss; 
	    ss << "[SiStripConfigDb::" << __func__ << "]" 
	       << " Vector of FED connections is empty for partition \"" 
	       << iter->second.partitionName()
	       << "\", therefore aborting upload for this partition!";
	    edm::LogWarning(mlConfigDb_) << ss.str(); 
	    continue; 
	  }

	} else {
	  // 	  stringstream ss; 
	  // 	  ss << "[SiStripConfigDb::" << __func__ << "]" 
	  // 	     << " Cannot find partition \"" << partition
	  // 	     << "\" in cached partitions list: \""
	  // 	     << dbParams_.partitionNames( dbParams_.partitionNames() ) 
	  // 	     << "\", therefore aborting upload for this partition!";
	  // 	  edm::LogWarning(mlConfigDb_) << ss.str(); 
	}

      }

    } catch (...) { handleException( __func__ ); }
    
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
    //edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
  // Reproduce temporary cache for "all partitions except specified one" (or clear all if none specified)
  FedConnections temporary_cache;
  if ( partition == ""  ) { temporary_cache = FedConnections(); }
  else {
    SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partitions().begin();
    SiStripDbParams::SiStripPartitions::const_iterator jter = dbParams_.partitions().end();
    for ( ; iter != jter; ++iter ) {
      if ( partition != iter->second.partitionName() ) {
	FedConnectionsRange range = connections_.find( iter->second.partitionName() );
	if ( range != connections_.emptyRange() ) {
	  temporary_cache.loadNext( partition, FedConnectionsV( range.begin(), range.end() ) );
	} else {
	  // 	  stringstream ss; 
	  // 	  ss << "[SiStripConfigDb::" << __func__ << "]" 
	  // 	     << " Cannot find partition \"" << iter->second.partitionName()
	  // 	     << "\" in local cache!";
	  // 	  edm::LogWarning(mlConfigDb_) << ss.str(); 
	}
      }
    }
  }

  // Delete objects in local cache for specified partition (or all if not specified) 
  FedConnectionsRange conns;
  if ( partition == "" ) { 
    if ( !connections_.empty() ) {
      conns = FedConnectionsRange( connections_.find( dbParams_.partitions().begin()->second.partitionName() ).begin(),
				   connections_.find( (--(dbParams_.partitions().end()))->second.partitionName() ).end() );
    } else { conns = connections_.emptyRange(); }
  } else {
    SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partitions().begin();
    SiStripDbParams::SiStripPartitions::const_iterator jter = dbParams_.partitions().end();
    for ( ; iter != jter; ++iter ) { if ( partition == iter->second.partitionName() ) { break; } }
    conns = connections_.find( iter->second.partitionName() );
  }
  
  if ( conns != connections_.emptyRange() ) {
    FedConnectionsV::const_iterator ifed = conns.begin();
    FedConnectionsV::const_iterator jfed = conns.end();
    for ( ; ifed != jfed; ++ifed ) { if ( *ifed ) { delete *ifed; } }
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
      ss << "  Partition name     : \"" << iconn->first << "\"" << std::endl;
      ss << "  Num of connections : " << iconn->second.size() << std::endl;

      // Extract FED ids and channels
      std::map< uint16_t, vector<uint16_t> > feds;
      FedConnectionsV::const_iterator iter = iconn->second.begin();
      FedConnectionsV::const_iterator jter = iconn->second.end();
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
