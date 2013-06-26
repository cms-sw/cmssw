// Last commit: $Id: FedDescriptions.cc,v 1.33 2011/09/02 11:25:25 eulisse Exp $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
SiStripConfigDb::FedDescriptionsRange SiStripConfigDb::getFedDescriptions( std::string partition ) {

  // Check
  if ( ( !dbParams_.usingDbCache() && !deviceFactory(__func__) ) ||
       (  dbParams_.usingDbCache() && !databaseCache(__func__) ) ) { 
    return feds_.emptyRange(); 
  }

  try {

    if ( !dbParams_.usingDbCache() ) { 

      SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partitions().begin();
      SiStripDbParams::SiStripPartitions::const_iterator jter = dbParams_.partitions().end();
      for ( ; iter != jter; ++iter ) {
	
	if ( partition == "" || partition == iter->second.partitionName() ) {

	  if ( iter->second.partitionName() == SiStripPartition::defaultPartitionName_ ) { continue; }

	  FedDescriptionsRange range = feds_.find( iter->second.partitionName() );
	  if ( range == feds_.emptyRange() ) {

	    // Extract versions
	    deviceFactory(__func__)->setUsingStrips( usingStrips_ );
	    int16_t major = iter->second.fedVersion().first; 
	    int16_t minor = iter->second.fedVersion().second; 
	    if ( iter->second.fedVersion().first == 0 && 
		 iter->second.fedVersion().second == 0 ) {
	      major = -1; //@@ "current state" for fed factory!
	      minor = -1; //@@ "current state" for fed factory!
	    }

	    // Retrive FED descriptions
	    FedDescriptionsV tmp1;
	    tmp1 = *( deviceFactory(__func__)->getFed9UDescriptions( iter->second.partitionName(), 
								     major, 
								     minor ) );
	    
	    // Make local copy 
	    FedDescriptionsV tmp2;
	    Fed9U::Fed9UDeviceFactory::vectorCopy( tmp2, tmp1 );
	    
	    // Add to cache
	    feds_.loadNext( iter->second.partitionName(), tmp2 );

	    // Some debug
	    FedDescriptionsRange feds = feds_.find( iter->second.partitionName() );
	    std::stringstream ss;
	    ss << "[SiStripConfigDb::" << __func__ << "]"
	       << " Downloaded " << feds.size() 
	       << " FED descriptions to local cache for partition \""
	       << iter->second.partitionName() << "\"" << std::endl;
	    ss << "[SiStripConfigDb::" << __func__ << "]"
	       << " Cache holds FED descriptions for " 
	       << feds_.size() << " partitions.";
	    LogTrace(mlConfigDb_) << ss.str();
	    
	  }

	}

      }
	    
    } else { // Using database cache

      FedDescriptionsV* tmp1 = databaseCache(__func__)->getFed9UDescriptions();
      
      if ( tmp1 ) { 
	
	// Make local copy 
	FedDescriptionsV tmp2;
	Fed9U::Fed9UDeviceFactory::vectorCopy( tmp2, *tmp1 );
	
	// Add to cache
	feds_.loadNext( SiStripPartition::defaultPartitionName_, tmp2 );
	
      } else {
	edm::LogWarning(mlConfigDb_)
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " NULL pointer to FED descriptions vector!";
      }

    }

  } catch (... ) { handleException( __func__ ); }
  
  // Create range object
  uint16_t np = 0;
  uint16_t nc = 0;
  FedDescriptionsRange feds;
  if ( partition != "" ) { 
    feds = feds_.find( partition );
    np = 1;
    nc = feds.size();
  } else {  
    if ( !feds_.empty() ) {
      feds = FedDescriptionsRange( feds_.find( dbParams_.partitions().begin()->second.partitionName() ).begin(),
				   feds_.find( (--(dbParams_.partitions().end()))->second.partitionName() ).end() );
    } else { feds = feds_.emptyRange(); }
    np = feds_.size();
    nc = feds.size();
  }
  
  stringstream ss; 
  ss << "[SiStripConfigDb::" << __func__ << "]"
     << " Found " << nc << " FED descriptions";
  if ( !dbParams_.usingDb() ) { ss << " in " << dbParams_.inputFedXmlFiles().size() << " 'fed.xml' file(s)"; }
  else { if ( !dbParams_.usingDbCache() )  { ss << " in " << np << " database partition(s)"; } 
  else { ss << " from shared memory name '" << dbParams_.sharedMemory() << "'"; } }
  if ( feds_.empty() ) { edm::LogWarning(mlConfigDb_) << ss.str(); }
  else { LogTrace(mlConfigDb_) << ss.str(); }
  
  return feds;
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::addFedDescriptions( std::string partition, FedDescriptionsV& feds ) {

  if ( !deviceFactory(__func__) ) { return; }

  if ( partition.empty() ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Partition string is empty,"
       << " therefore cannot add FED descriptions to local cache!"; 
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
  if ( feds.empty() ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Vector of FED descriptions is empty,"
       << " therefore cannot add FED descriptions to local cache!"; 
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
       << " therefore cannot add FED descriptions!";
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
  FedDescriptionsRange range = feds_.find( partition );
  if ( range == feds_.emptyRange() ) {
    
    // Make local copy 
    FedDescriptionsV tmp;
    Fed9U::Fed9UDeviceFactory::vectorCopy( tmp, feds );

    // Add to local cache
    feds_.loadNext( partition, tmp );
    
    // Some debug
    std::stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " Added " << feds.size() 
       << " FED descriptions to local cache for partition \""
       << iter->second.partitionName() << "\"" << std::endl;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " Cache holds FED descriptions for " 
       << feds_.size() << " partitions.";
    LogTrace(mlConfigDb_) << ss.str();
    
  } else {
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Partition \"" << partition
       << "\" already found in local cache, "
       << " therefore cannot add new FED descriptions!";
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::uploadFedDescriptions( std::string partition ) { 
  
  if ( dbParams_.usingDbCache() ) {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]" 
      << " Using database cache! No uploads allowed!"; 
    return;
  }
  
  if ( !deviceFactory(__func__) ) { return; }
  
  if ( feds_.empty() ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Found no cached FED descriptions, therefore no upload!"; 
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
  try { 

    SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partitions().begin();
    SiStripDbParams::SiStripPartitions::const_iterator jter = dbParams_.partitions().end();
    for ( ; iter != jter; ++iter ) {
      
      if ( partition == "" || partition == iter->second.partitionName() ) {
	
	FedDescriptionsRange range = feds_.find( iter->second.partitionName() );
	if ( range != feds_.emptyRange() ) {
	  
	  FedDescriptionsV feds( range.begin(), range.end() );
	  
          SiStripPartition::Versions fedVersion = iter->second.fedVersion();
	  deviceFactory(__func__)->setFed9UDescriptions( feds,
							 iter->second.partitionName(),
							 (uint16_t*)(&(fedVersion.first)), 
							 (uint16_t*)(&(fedVersion.second)),
							 1 ); // new major version

	  // Some debug
	  std::stringstream ss;
	  ss << "[SiStripConfigDb::" << __func__ << "]"
	     << " Uploaded " << feds.size() 
	     << " FED descriptions to database for partition \""
	     << iter->second.partitionName() << "\"";
	  LogTrace(mlConfigDb_) << ss.str();
	  
	} else {
	  stringstream ss; 
	  ss << "[SiStripConfigDb::" << __func__ << "]" 
	     << " Vector of FED descriptions is empty for partition \"" 
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
  
  allowCalibUpload_ = true;
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::clearFedDescriptions( std::string partition ) {
  LogTrace(mlConfigDb_) << "[SiStripConfigDb::" << __func__ << "]";
  
  if ( feds_.empty() ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Found no cached FED descriptions!"; 
    //edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
  // Reproduce temporary cache for "all partitions except specified one" (or clear all if none specified)
  FedDescriptions temporary_cache;
  if ( partition == ""  ) { temporary_cache = FedDescriptions(); }
  else {
    SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partitions().begin();
    SiStripDbParams::SiStripPartitions::const_iterator jter = dbParams_.partitions().end();
    for ( ; iter != jter; ++iter ) {
      if ( partition != iter->second.partitionName() ) {
	FedDescriptionsRange range = feds_.find( iter->second.partitionName() );
	if ( range != feds_.emptyRange() ) {
	  temporary_cache.loadNext( partition, FedDescriptionsV( range.begin(), range.end() ) );
	}
      } else {
	FedDescriptionsRange range = feds_.find( iter->second.partitionName() );
	if ( range != feds_.emptyRange() ) {
	  LogTrace(mlConfigDb_) 
	    << "[SiStripConfigDb::" << __func__ << "]"
	    << " Deleting FED descriptions for partition \""
	    << iter->second.partitionName()
	    << "\" from local cache...";
	}
      }
    }
  }
  
  // Delete objects in local cache for specified partition (or all if not specified) 
  FedDescriptionsRange feds;
  if ( partition == "" ) { 
    if ( !feds_.empty() ) {
      feds = FedDescriptionsRange( feds_.find( dbParams_.partitions().begin()->second.partitionName() ).begin(),
				   feds_.find( (--(dbParams_.partitions().end()))->second.partitionName() ).end() );
    } else { feds = feds_.emptyRange(); }
  } else {
    SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partitions().begin();
    SiStripDbParams::SiStripPartitions::const_iterator jter = dbParams_.partitions().end();
    for ( ; iter != jter; ++iter ) { if ( partition == iter->second.partitionName() ) { break; } }
    feds = feds_.find( iter->second.partitionName() );
  }
  
  if ( feds != feds_.emptyRange() ) {
    FedDescriptionsV::const_iterator ifed = feds.begin();
    FedDescriptionsV::const_iterator jfed = feds.end();
    for ( ; ifed != jfed; ++ifed ) { if ( *ifed ) { delete *ifed; } }
  } else {
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]";
    if ( partition == "" ) { ss << " Found no FED descriptions in local cache!"; }
    else { ss << " Found no FED descriptions in local cache for partition \"" << partition << "\"!"; }
    edm::LogWarning(mlConfigDb_) << ss.str(); 
  }
  
  // Overwrite local cache with temporary cache
  feds_ = temporary_cache; 

}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::printFedDescriptions( std::string partition ) {

  std::stringstream ss;
  ss << "[SiStripConfigDb::" << __func__ << "]"
     << " Contents of FedDescriptions container:" << std::endl;
  ss << " Number of partitions: " << feds_.size() << std::endl;

  // Loop through partitions
  uint16_t cntr = 0;
  FedDescriptions::const_iterator iconn = feds_.begin();
  FedDescriptions::const_iterator jconn = feds_.end();
  for ( ; iconn != jconn; ++iconn ) {

    cntr++;
    if ( partition == "" || partition == iconn->first ) {
      
      ss << "  Partition number : " << cntr << " (out of " << feds_.size() << ")" << std::endl;
      ss << "  Partition name   : \"" << iconn->first << "\"" << std::endl;
      ss << "  Num of FED ids   : " << iconn->second.size() << std::endl;

      // Extract FED crates and ids
      std::map< uint16_t, vector<uint16_t> > feds;
      FedDescriptionsV::const_iterator iter = iconn->second.begin();
      FedDescriptionsV::const_iterator jter = iconn->second.end();
      for ( ; iter != jter; ++iter ) { 
	if ( *iter ) { 
	  uint16_t key = (*iter)->getCrateNumber();
	  uint16_t data = (*iter)->getFedId();
	  if ( find( feds[key].begin(), feds[key].end(), data ) == feds[key].end() ) { 
	    feds[key].push_back( data ); 
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
      
      // Print FED crates and ids
      std::map< uint16_t, std::vector<uint16_t> >::const_iterator ifed = feds.begin();
      std::map< uint16_t, std::vector<uint16_t> >::const_iterator jfed = feds.end();
      for ( ; ifed != jfed; ++ifed ) {
	ss << "  Found " << std::setw(2) << ifed->second.size()
	   << " FED ids for crate number " << std::setw(2) << ifed->first << " : ";
	if ( !ifed->second.empty() ) { 
	  uint16_t first = ifed->second.front();
	  uint16_t last = ifed->second.front();
	  std::vector<uint16_t>::const_iterator icrate = ifed->second.begin();
	  std::vector<uint16_t>::const_iterator jcrate = ifed->second.end();
	  for ( ; icrate != jcrate; ++icrate ) { 
	    if ( icrate != ifed->second.begin() ) {
	      if ( *icrate != last+1 ) { 
		ss << std::setw(2) << first << "->" << std::setw(2) << last << ", ";
		if ( icrate != ifed->second.end() ) { first = *(icrate+1); }
	      } 
	    }
	    last = *icrate;
	  }
	  if ( first != last ) { ss << std::setw(2) << first << "->" << std::setw(2) << last; }
	  ss << std::endl;
	}
      }
      
    }
    
  }
  
  LogTrace(mlConfigDb_) << ss.str();

}

// -----------------------------------------------------------------------------
/** */ 
SiStripConfigDb::FedIdsRange SiStripConfigDb::getFedIds( std::string partition ) {
  
  fedIds_.clear();
  
  if ( ( !dbParams_.usingDbCache() && !deviceFactory(__func__) ) ||
       (  dbParams_.usingDbCache() && !databaseCache(__func__) ) ) { 
    return FedIdsRange( fedIds_.end(), fedIds_.end() );
  }
  
  try { 

    // Inhibit download of strip info
    bool using_strips = usingStrips_;
    if ( factory_ ) { factory_->setUsingStrips( false ); }
    FedDescriptionsRange feds = getFedDescriptions( partition );
    if ( factory_ ) { factory_->setUsingStrips( using_strips ); }
    
    if ( !feds.empty() ) {
      FedDescriptionsV::const_iterator ifed = feds.begin();
      FedDescriptionsV::const_iterator jfed = feds.end();
      for ( ; ifed != jfed; ++ifed ) { 
	if ( *ifed ) { fedIds_.push_back( (*ifed)->getFedId() ); }
	else {
	  edm::LogError(mlCabling_)
	    << "[SiStripConfigDb::" << __func__ << "]"
	    << " NULL pointer to FedDescription!";
	  continue;
	}
      }
    }
    
  } catch (...) { handleException( __func__ ); }
  
  if ( fedIds_.empty() ) {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " No FED ids found!"; 
  }
  
  return FedIdsRange( fedIds_.begin(), fedIds_.end() );

}

