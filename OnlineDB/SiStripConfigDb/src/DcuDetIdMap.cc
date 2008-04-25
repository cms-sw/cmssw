// Last commit: $Id: DcuDetIdMap.cc,v 1.17 2008/04/24 16:02:34 bainbrid Exp $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
SiStripConfigDb::DcuDetIdMap::range SiStripConfigDb::getDcuDetIdMap( std::string partition ) {

  // Check
  if ( ( !dbParams_.usingDbCache_ && !deviceFactory(__func__) ) ||
       (  dbParams_.usingDbCache_ && !databaseCache(__func__) ) ) { 
    return dcuDetIdMap_.emptyRange(); 
  }
  
  try {

    if ( !dbParams_.usingDbCache_ ) { 

      SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partitions_.begin();
      SiStripDbParams::SiStripPartitions::const_iterator jter = dbParams_.partitions_.end();
      for ( ; iter != jter; ++iter ) {
	
	if ( partition == "" || partition == iter->second.partitionName() ) {
	  
	  DcuDetIdMap::range range = dcuDetIdMap_.find( iter->second.partitionName() );
	  if ( range == dcuDetIdMap_.emptyRange() ) {
	    
#ifdef USING_NEW_DATABASE_MODEL
	    deviceFactory(__func__)->addDetIdPartition( iter->second.partitionName(),
							iter->second.dcuVersion().first, 
							iter->second.dcuVersion().second );
#else
	    deviceFactory(__func__)->addDetIdPartition( iter->second.partitionName() );
#endif
	    
	    // Retrieve DCU-DetId map
	    HashMap src = deviceFactory(__func__)->getInfos(); 
	    
	    // Make local copy 
	    std::vector<DcuDetId> dst;
	    clone( src, dst );
	    
	    // Add to cache
	    dcuDetIdMap_.loadNext( iter->second.partitionName(), dst );
	    
	  }

	}

      }
      
    } else {
      
#ifdef USING_NEW_DATABASE_MODEL
      
      // Retrieve DCU-DetId map
      HashMap* src = databaseCache(__func__)->getInfos(); 

      if ( src ) { 
	
	// Make local copy 
	std::vector<DcuDetId> dst;
	clone( *src, dst ); 
	
	// Add to cache
	dcuDetIdMap_.loadNext( "", dst );
	
      } else {
	edm::LogWarning(mlConfigDb_)
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " NULL pointer to Dcu-DetId map!";
      }
#endif
      
    }
    
  } catch (... ) { handleException( __func__ ); }

  // Create range object
  uint16_t np = 0;
  uint16_t nc = 0;
  DcuDetIdMap::range range;
  if ( partition != "" ) { 
    range = dcuDetIdMap_.find( partition );
    np = 1;
    nc = range.size();
  } else { 
    range = DcuDetIdMap::range( dcuDetIdMap_.find( dbParams_.partitions_.begin()->second.partitionName() ).begin(),
				dcuDetIdMap_.find( dbParams_.partitions_.rbegin()->second.partitionName() ).end() );
    np = dcuDetIdMap_.size();
    nc = range.size();
  }
  
  stringstream ss; 
  ss << "[SiStripConfigDb::" << __func__ << "]"
     << " Found " << nc << " FED connections";
  if ( !dbParams_.usingDb_ ) { ss << " in " << dbParams_.inputDcuInfoXmlFiles().size() << " 'module.xml' file(s)"; }
  else { if ( !dbParams_.usingDbCache_ )  { ss << " in " << np << " database partition(s)"; } 
  else { ss << " from shared memory name '" << dbParams_.sharedMemory_ << "'"; } }
  if ( dcuDetIdMap_.empty() ) { edm::LogWarning(mlConfigDb_) << ss.str(); }
  else { LogTrace(mlConfigDb_) << ss.str(); }

  return range;
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::addDcuDetIdMap( std::string partition, std::vector<DcuDetId>& dcus ) {
  
  if ( !deviceFactory(__func__) ) { return; }
  
  if ( partition.empty() ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Partition string is empty,"
       << " therefore cannot add DCU-DetId map to local cache!"; 
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
  if ( dcus.empty() ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Vector of DCU-DetId map is empty,"
       << " therefore cannot add DCU-DetId map to local cache!"; 
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }

  SiStripDbParams::SiStripPartitions::iterator iter = dbParams_.partitions_.begin();
  SiStripDbParams::SiStripPartitions::iterator jter = dbParams_.partitions_.end();
  for ( ; iter != jter; ++iter ) { if ( partition == iter->second.partitionName() ) { break; } }
  if ( iter == dbParams_.partitions_.end() ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Partition \"" << partition
       << "\" not found in partition list, "
       << " therefore cannot add DCU-DetId map!";
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
  DcuDetIdMap::range range = dcuDetIdMap_.find( partition );
  if ( range == dcuDetIdMap_.emptyRange() ) {
    
    // Make local copy 
    std::vector<DcuDetId> dst;
#ifdef USING_NEW_DATABASE_MODEL
    clone( dcus, dst );
#else
    dst = dcus;
#endif
    
    // Add to local cache
    dcuDetIdMap_.loadNext( partition, dst );
    
    // Some debug
    std::stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " Added " << dst.size() 
       << " DCU-DetId map to local cache for partition \""
       << partition << "\"."
       << " (Cache holds DCU-DetId map for " 
       << dcuDetIdMap_.size() << " partitions.)";
    LogTrace(mlConfigDb_) << ss.str();
    
  } else {
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Partition \"" << partition
       << "\" already found in local cache, "
       << " therefore cannot add new DCU-DetId map!";
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::uploadDcuDetIdMap( std::string partition ) {
  LogTrace(mlConfigDb_) << "[SiStripConfigDb::" << __func__ << "]";
  
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
  
//   if ( dbParams_.usingDbCache_ ) {
//     edm::LogWarning(mlConfigDb_)
//       << "[SiStripConfigDb::" << __func__ << "]" 
//       << " Using database cache! No uploads allowed!"; 
//     return;
//   }

//   if ( !deviceFactory(__func__) ) { return; }
  
//   if ( dcuDetIdMap_.empty() ) { 
//     stringstream ss; 
//     ss << "[SiStripConfigDb::" << __func__ << "]" 
//        << " Found no cached DCU-DetId map, therefore no upload!"; 
//     edm::LogWarning(mlConfigDb_) << ss.str(); 
//     return; 
//   }

//   try {
    
//     SiStripDbParams::SiStripPartitions::iterator iter = dbParams_.partitions_.begin();
//     SiStripDbParams::SiStripPartitions::iterator jter = dbParams_.partitions_.end();
//     for ( ; iter != jter; ++iter ) {
      
//       if ( partition == "" || partition == iter->second.partitionName() ) {
	
// 	DcuDetIdMap::range range = dcuDetIdMap_.find( iter->second.partitionName() );
// 	if ( range != dcuDetIdMap_.emptyRange() ) {
	  
// 	  // Extract 
// 	  HashMap dst;
// 	  clone( std::vector<DcuDetId>( range.begin(), range.end() ), dst );
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
// 	// 	     << dbParams_.partitions( dbParams_.partitions() ) 
// 	// 	     << "\", therefore aborting upload for this partition!";
// 	// 	  edm::LogWarning(mlConfigDb_) << ss.str(); 
//       }
      
//     }
    
//   } catch (... ) { handleException( __func__, "Problems updating objects in TkDcuInfoFactory!" ); }
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::clearDcuDetIdMap( std::string partition ) {
  LogTrace(mlConfigDb_) << "[SiStripConfigDb::" << __func__ << "]";
  
  if ( dcuDetIdMap_.empty() ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Found no cached DCU-DetId map!"; 
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
  // Reproduce temporary cache for "all partitions except specified one" (or clear all if none specified)
  DcuDetIdMap temporary_cache;
  if ( partition == ""  ) { temporary_cache = DcuDetIdMap(); }
  else {
    SiStripDbParams::SiStripPartitions::iterator iter = dbParams_.partitions_.begin();
    SiStripDbParams::SiStripPartitions::iterator jter = dbParams_.partitions_.end();
    for ( ; iter != jter; ++iter ) {
      if ( partition != iter->second.partitionName() ) {
	DcuDetIdMap::range range = dcuDetIdMap_.find( iter->second.partitionName() );
	if ( range != dcuDetIdMap_.emptyRange() ) {
	  temporary_cache.loadNext( partition, std::vector<DcuDetId>( range.begin(), range.end() ) );
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
  DcuDetIdMap::range dcus;
  if ( partition == "" ) { 
    dcus = DcuDetIdMap::range( dcuDetIdMap_.find( dbParams_.partitions_.begin()->second.partitionName() ).begin(),
			       dcuDetIdMap_.find( dbParams_.partitions_.rbegin()->second.partitionName() ).end() );
  } else {
    SiStripDbParams::SiStripPartitions::iterator iter = dbParams_.partitions_.begin();
    SiStripDbParams::SiStripPartitions::iterator jter = dbParams_.partitions_.end();
    for ( ; iter != jter; ++iter ) { if ( partition == iter->second.partitionName() ) { break; } }
    dcus = dcuDetIdMap_.find( iter->second.partitionName() );
  }
  
  if ( dcus != dcuDetIdMap_.emptyRange() ) {
    
#ifdef USING_NEW_DATABASE_MODEL	
    std::vector<DcuDetId>::const_iterator ifed = dcus.begin();
    std::vector<DcuDetId>::const_iterator jfed = dcus.end();
    for ( ; ifed != jfed; ++ifed ) { if ( ifed->second ) { delete ifed->second; } }
#endif
    
  } else {
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]";
    if ( partition == "" ) { ss << " Found no DCU-DetId map in local cache!"; }
    else { ss << " Found no DCU-DetId map in local cache for partition \"" << partition << "\"!"; }
    edm::LogWarning(mlConfigDb_) << ss.str(); 
  }
  
  // Overwrite local cache with temporary cache
  dcuDetIdMap_ = temporary_cache; 

}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::printDcuDetIdMap( std::string partition ) {

  std::stringstream ss;
  ss << "[SiStripConfigDb::" << __func__ << "]"
     << " Contents of DcuDetIdMap container:" << std::endl;
  ss << " Number of partitions: " << dcuDetIdMap_.size() << std::endl;

  // Loop through partitions
  uint16_t cntr = 0;
  DcuDetIdMap::const_iterator iconn = dcuDetIdMap_.begin();
  DcuDetIdMap::const_iterator jconn = dcuDetIdMap_.end();
  for ( ; iconn != jconn; ++iconn ) {

    cntr++;
    if ( partition == "" || partition == iconn->first ) {
      
      ss << "  Partition number      : " << cntr << " (out of " << dcuDetIdMap_.size() << ")" << std::endl;
      ss << "  Partition name        : " << iconn->first << std::endl;
      ss << "  Size of DCU-DetId map : " << iconn->second.size() << std::endl;
      
    }
    
  }
  
  LogTrace(mlConfigDb_) << ss.str();
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::clone( const HashMap& input, std::vector<DcuDetId>& output ) const {
  output.clear();
  HashMap::const_iterator ii = input.begin();
  HashMap::const_iterator jj = input.end();
  for ( ; ii != jj; ++ii ) { if ( ii->second ) { output.push_back( std::make_pair( ii->first, new TkDcuInfo( *(ii->second) ) ) ); } }
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::clone( const std::vector<DcuDetId>& input, HashMap& output ) const {
  output.clear();
  std::vector<DcuDetId>::const_iterator ii = input.begin();
  std::vector<DcuDetId>::const_iterator jj = input.end();
  for ( ; ii != jj; ++ii ) { if ( ii->second ) { output[ii->first] = new TkDcuInfo( *(ii->second) ); } }
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::clone( const std::vector<DcuDetId>& input, std::vector<DcuDetId>& output ) const {
  output.clear();
  std::vector<DcuDetId>::const_iterator ii = input.begin();
  std::vector<DcuDetId>::const_iterator jj = input.end();
  for ( ; ii != jj; ++ii ) { if ( ii->second ) { output.push_back( std::make_pair( ii->first, new TkDcuInfo( *(ii->second) ) ) ); } }
}
