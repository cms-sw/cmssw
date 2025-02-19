// Last commit: $Id: DcuDetIds.cc,v 1.7 2009/04/06 16:57:28 lowette Exp $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
SiStripConfigDb::DcuDetIdsRange SiStripConfigDb::getDcuDetIds( std::string partition ) {

  // Check
  if ( ( !dbParams_.usingDbCache() && !deviceFactory(__func__) ) ||
       (  dbParams_.usingDbCache() && !databaseCache(__func__) ) ) { 
    return dcuDetIds_.emptyRange(); 
  }
  
  try {

    if ( !dbParams_.usingDbCache() ) { 

      SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partitions().begin();
      SiStripDbParams::SiStripPartitions::const_iterator jter = dbParams_.partitions().end();
      for ( ; iter != jter; ++iter ) {
	
	if ( partition == "" || partition == iter->second.partitionName() ) {
	  
	  if ( iter->second.partitionName() == SiStripPartition::defaultPartitionName_ ) { continue; }

	  DcuDetIdsRange range = dcuDetIds_.find( iter->second.partitionName() );
	  if ( range == dcuDetIds_.emptyRange() ) {
	    
	    deviceFactory(__func__)->addDetIdPartition( iter->second.partitionName(),
							iter->second.dcuVersion().first, 
							iter->second.dcuVersion().second );
	    
	    // Retrieve DCU-DetId map
	    DcuDetIdMap src = deviceFactory(__func__)->getInfos(); 
	    
	    // Make local copy 
	    DcuDetIdsV dst;
	    clone( src, dst );
	    
	    // Add to cache
	    dcuDetIds_.loadNext( iter->second.partitionName(), dst );
	    
	  }

	}

      }
      
    } else { // Using database cache
      
      // Retrieve DCU-DetId map
      DcuDetIdMap* src = databaseCache(__func__)->getInfos(); 

      if ( src ) { 
	
	// Make local copy 
	DcuDetIdsV dst;
	clone( *src, dst ); 
	
	// Add to cache
	dcuDetIds_.loadNext( SiStripPartition::defaultPartitionName_, dst );
	
      } else {
	edm::LogWarning(mlConfigDb_)
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " NULL pointer to Dcu-DetId map!";
      }
      
    }
    
  } catch (... ) { handleException( __func__ ); }

  // Create range object
  uint16_t np = 0;
  uint16_t nc = 0;
  DcuDetIdsRange range;
  if ( partition != "" ) { 
    range = dcuDetIds_.find( partition );
    np = 1;
    nc = range.size();
  } else { 
    if ( !dcuDetIds_.empty() ) {
      range = DcuDetIdsRange( dcuDetIds_.find( dbParams_.partitions().begin()->second.partitionName() ).begin(),
			      dcuDetIds_.find( (--(dbParams_.partitions().end()))->second.partitionName() ).end() );
    } else { range = dcuDetIds_.emptyRange(); }
    np = dcuDetIds_.size();
    nc = range.size();
  }
  
  stringstream ss; 
  ss << "[SiStripConfigDb::" << __func__ << "]"
     << " Found " << nc << " entries in DCU-DetId map";
  if ( !dbParams_.usingDb() ) { ss << " in " << dbParams_.inputDcuInfoXmlFiles().size() << " 'module.xml' file(s)"; }
  else { if ( !dbParams_.usingDbCache() )  { ss << " in " << np << " database partition(s)"; } 
  else { ss << " from shared memory name '" << dbParams_.sharedMemory() << "'"; } }
  if ( dcuDetIds_.empty() ) { edm::LogWarning(mlConfigDb_) << ss.str(); }
  else { LogTrace(mlConfigDb_) << ss.str(); }

  return range;
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::addDcuDetIds( std::string partition, DcuDetIdsV& dcus ) {

  stringstream ss; 
  ss << "[SiStripConfigDb::" << __func__ << "]" 
     << " Cannot add to local cache! This functionality not allowed!";
  edm::LogWarning(mlConfigDb_) << ss.str(); 
  
  //   if ( !deviceFactory(__func__) ) { return; }
  
  //   if ( partition.empty() ) { 
  //     stringstream ss; 
  //     ss << "[SiStripConfigDb::" << __func__ << "]" 
  //        << " Partition string is empty,"
  //        << " therefore cannot add DCU-DetId map to local cache!"; 
  //     edm::LogWarning(mlConfigDb_) << ss.str(); 
  //     return; 
  //   }
  
  //   if ( dcus.empty() ) { 
  //     stringstream ss; 
  //     ss << "[SiStripConfigDb::" << __func__ << "]" 
  //        << " Vector of DCU-DetId map is empty,"
  //        << " therefore cannot add DCU-DetId map to local cache!"; 
  //     edm::LogWarning(mlConfigDb_) << ss.str(); 
  //     return; 
  //   }

  //   SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partitions().begin();
  //   SiStripDbParams::SiStripPartitions::const_iterator jter = dbParams_.partitions().end();
  //   for ( ; iter != jter; ++iter ) { if ( partition == iter->second.partitionName() ) { break; } }
  //   if ( iter == dbParams_.partitions().end() ) { 
  //     stringstream ss; 
  //     ss << "[SiStripConfigDb::" << __func__ << "]" 
  //        << " Partition \"" << partition
  //        << "\" not found in partition list, "
  //        << " therefore cannot add DCU-DetId map!";
  //     edm::LogWarning(mlConfigDb_) << ss.str(); 
  //     return; 
  //   }
  
  //   DcuDetIdsRange range = dcuDetIds_.find( partition );
  //   if ( range == dcuDetIds_.emptyRange() ) {
    
  //     // Make local copy 
  //     DcuDetIdsV dst;
  //     clone( dcus, dst );
    
  //     // Add to local cache
  //     dcuDetIds_.loadNext( partition, dst );
    
  //     // Some debug
  //     std::stringstream ss;
  //     ss << "[SiStripConfigDb::" << __func__ << "]"
  //        << " Added " << dst.size() 
  //        << " DCU-DetId map to local cache for partition \""
  //        << partition << "\"" << std::endl;
  //     ss << "[SiStripConfigDb::" << __func__ << "]"
  //        << " Cache holds DCU-DetId map for " 
  //        << dcuDetIds_.size() << " partitions.";
  //     LogTrace(mlConfigDb_) << ss.str();
    
  //   } else {
  //     stringstream ss; 
  //     ss << "[SiStripConfigDb::" << __func__ << "]" 
  //        << " Partition \"" << partition
  //        << "\" already found in local cache, "
  //        << " therefore cannot add new DCU-DetId map!";
  //     edm::LogWarning(mlConfigDb_) << ss.str(); 
  //     return; 
  //   }
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::uploadDcuDetIds( std::string partition ) {

  stringstream ss; 
  ss << "[SiStripConfigDb::" << __func__ << "]" 
     << " Cannot upload to database! This functionality not allowed!";
  edm::LogWarning(mlConfigDb_) << ss.str(); 
  
  /*
    addAllDetId => all detids
    addAllDetId => to download (up to you)
    change in the detids
    setTkDcuInfo
    getCurrentStates
    setCurrentState
    addDetIpartiton
    AddAllDetId
  */
  
  //   if ( dbParams_.usingDbCache() ) {
  //     edm::LogWarning(mlConfigDb_)
  //       << "[SiStripConfigDb::" << __func__ << "]" 
  //       << " Using database cache! No uploads allowed!"; 
  //     return;
  //   }

  //   if ( !deviceFactory(__func__) ) { return; }
  
  //   if ( dcuDetIds_.empty() ) { 
  //     stringstream ss; 
  //     ss << "[SiStripConfigDb::" << __func__ << "]" 
  //        << " Found no cached DCU-DetId map, therefore no upload!"; 
  //     edm::LogWarning(mlConfigDb_) << ss.str(); 
  //     return; 
  //   }

  //   try {
    
  //     SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partitions().begin();
  //     SiStripDbParams::SiStripPartitions::const_iterator jter = dbParams_.partitions().end();
  //     for ( ; iter != jter; ++iter ) {
      
  //       if ( partition == "" || partition == iter->second.partitionName() ) {
	
  // 	DcuDetIdsRange range = dcuDetIds_.find( iter->second.partitionName() );
  // 	if ( range != dcuDetIds_.emptyRange() ) {
	  
  // 	  // Extract 
  // 	  DcuDetIdMap dst;
  // 	  clone( DcuDetIdsV( range.begin(), range.end() ), dst );
  // 	  deviceFactory(__func__)->setTkDcuInfo( dst );
  // 	  getcurrentstate
  // 	  deviceFactory(__func__)->addAllDetId();
	  
  // 	  // Some debug
  // 	  std::stringstream ss;
  // 	  ss << "[SiStripConfigDb::" << __func__ << "]"
  // 	     << " Uploaded " << dst.size() 
  // 	     << " DCU-DetId map to DB/xml for partition \""
  // 	     << iter->second.partitionName() << "\".";
  // 	  LogTrace(mlConfigDb_) << ss.str();
	  
  // 	} else {
  // 	  stringstream ss; 
  // 	  ss << "[SiStripConfigDb::" << __func__ << "]" 
  // 	     << " Vector of DCU-DetId map is empty for partition \"" 
  // 	     << iter->second.partitionName()
  // 	     << "\", therefore aborting upload for this partition!";
  // 	  edm::LogWarning(mlConfigDb_) << ss.str(); 
  // 	  continue; 
  // 	}
	
  //       } else {
  // 	// 	  stringstream ss; 
  // 	// 	  ss << "[SiStripConfigDb::" << __func__ << "]" 
  // 	// 	     << " Cannot find partition \"" << partition
  // 	// 	     << "\" in cached partitions list: \""
  // 	// 	     << dbParams_.partitionNames( dbParams_.partitionNames() ) 
  // 	// 	     << "\", therefore aborting upload for this partition!";
  // 	// 	  edm::LogWarning(mlConfigDb_) << ss.str(); 
  //       }
      
  //     }
    
  //   } catch (... ) { handleException( __func__, "Problems updating objects in TkDcuInfoFactory!" ); }
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::clearDcuDetIds( std::string partition ) {
  LogTrace(mlConfigDb_) << "[SiStripConfigDb::" << __func__ << "]";
  
  if ( dcuDetIds_.empty() ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Found no cached DCU-DetId map!"; 
    //edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
  // Reproduce temporary cache for "all partitions except specified one" (or clear all if none specified)
  DcuDetIds temporary_cache;
  if ( partition == ""  ) { temporary_cache = DcuDetIds(); }
  else {
    SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partitions().begin();
    SiStripDbParams::SiStripPartitions::const_iterator jter = dbParams_.partitions().end();
    for ( ; iter != jter; ++iter ) {
      if ( partition != iter->second.partitionName() ) {
	DcuDetIdsRange range = dcuDetIds_.find( iter->second.partitionName() );
	if ( range != dcuDetIds_.emptyRange() ) {
	  temporary_cache.loadNext( partition, DcuDetIdsV( range.begin(), range.end() ) );
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
  DcuDetIdsRange dcus;
  if ( partition == "" ) { 
    if ( !dcuDetIds_.empty() ) {
      dcus = DcuDetIdsRange( dcuDetIds_.find( dbParams_.partitions().begin()->second.partitionName() ).begin(),
			     dcuDetIds_.find( (--(dbParams_.partitions().end()))->second.partitionName() ).end() );
    } else { dcus = dcuDetIds_.emptyRange(); }
  } else {
    SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partitions().begin();
    SiStripDbParams::SiStripPartitions::const_iterator jter = dbParams_.partitions().end();
    for ( ; iter != jter; ++iter ) { if ( partition == iter->second.partitionName() ) { break; } }
    dcus = dcuDetIds_.find( iter->second.partitionName() );
  }
  
  if ( dcus != dcuDetIds_.emptyRange() ) {
    DcuDetIdsV::const_iterator ifed = dcus.begin();
    DcuDetIdsV::const_iterator jfed = dcus.end();
    for ( ; ifed != jfed; ++ifed ) { if ( ifed->second ) { delete ifed->second; } }
  } else {
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]";
    if ( partition == "" ) { ss << " Found no DCU-DetId map in local cache!"; }
    else { ss << " Found no DCU-DetId map in local cache for partition \"" << partition << "\"!"; }
    edm::LogWarning(mlConfigDb_) << ss.str(); 
  }
  
  // Overwrite local cache with temporary cache
  dcuDetIds_ = temporary_cache; 

}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::printDcuDetIds( std::string partition ) {

  std::stringstream ss;
  ss << "[SiStripConfigDb::" << __func__ << "]"
     << " Contents of DcuDetIds container:" << std::endl;
  ss << " Number of partitions: " << dcuDetIds_.size() << std::endl;

  // Loop through partitions
  uint16_t cntr = 0;
  DcuDetIds::const_iterator idcu = dcuDetIds_.begin();
  DcuDetIds::const_iterator jdcu = dcuDetIds_.end();
  for ( ; idcu != jdcu; ++idcu ) {

    cntr++;
    if ( partition == "" || partition == idcu->first ) {
      
      ss << "  Partition number      : " << cntr << " (out of " << dcuDetIds_.size() << ")" << std::endl;
      ss << "  Partition name        : \"" << idcu->first << "\"" << std::endl;
      ss << "  Size of DCU-DetId map : " << idcu->second.size() << std::endl;
      
    }
    
  }
  
  LogTrace(mlConfigDb_) << ss.str();
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::clone( const DcuDetIdMap& input, DcuDetIdsV& output ) const {
  output.clear();
  DcuDetIdMap::const_iterator ii = input.begin();
  DcuDetIdMap::const_iterator jj = input.end();
  for ( ; ii != jj; ++ii ) { if ( ii->second ) { output.push_back( std::make_pair( ii->first, new TkDcuInfo( *(ii->second) ) ) ); } }
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::clone( const DcuDetIdsV& input, DcuDetIdMap& output ) const {
  output.clear();
  DcuDetIdsV::const_iterator ii = input.begin();
  DcuDetIdsV::const_iterator jj = input.end();
  for ( ; ii != jj; ++ii ) { if ( ii->second ) { output[ii->first] = new TkDcuInfo( *(ii->second) ); } }
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::clone( const DcuDetIdsV& input, DcuDetIdsV& output ) const {
  output.clear();
  DcuDetIdsV::const_iterator ii = input.begin();
  DcuDetIdsV::const_iterator jj = input.end();
  for ( ; ii != jj; ++ii ) { if ( ii->second ) { output.push_back( std::make_pair( ii->first, new TkDcuInfo( *(ii->second) ) ) ); } }
}

// -----------------------------------------------------------------------------
// 
SiStripConfigDb::DcuDetIdsV::const_iterator SiStripConfigDb::findDcuDetId( DcuDetIdsV::const_iterator begin, 
									   DcuDetIdsV::const_iterator end, 
									   uint32_t dcu_id  ) {
  DcuDetIdsV::const_iterator iter = begin;
  DcuDetIdsV::const_iterator jter = end;
  for ( ; iter != jter; ++iter  ) {
    if ( iter->second && iter->second->getDcuHardId() == dcu_id ) { return iter; }
  }
  return end;
}

// -----------------------------------------------------------------------------
// 
SiStripConfigDb::DcuDetIdsV::iterator SiStripConfigDb::findDcuDetId( DcuDetIdsV::iterator begin, 
								     DcuDetIdsV::iterator end, 
								     uint32_t dcu_id  ) {
  DcuDetIdsV::iterator iter = begin;
  DcuDetIdsV::iterator jter = end;
  for ( ; iter != jter; ++iter  ) {
    if ( iter->second && iter->second->getDcuHardId() == dcu_id ) { return iter; }
  }
  return end;
}
