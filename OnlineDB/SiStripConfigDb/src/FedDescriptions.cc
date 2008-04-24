// Last commit: $Id: FedDescriptions.cc,v 1.22 2008/04/21 09:52:41 bainbrid Exp $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
SiStripConfigDb::FedDescriptions::range SiStripConfigDb::getFedDescriptions( std::string partition ) {

  // Check
  if ( ( !dbParams_.usingDbCache_ && !deviceFactory(__func__) ) ||
       (  dbParams_.usingDbCache_ && !databaseCache(__func__) ) ) { 
    return feds_.emptyRange(); 
  }

  try {

    if ( !dbParams_.usingDbCache_ ) { 

      SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partitions_.begin();
      SiStripDbParams::SiStripPartitions::const_iterator jter = dbParams_.partitions_.end();
      for ( ; iter != jter; ++iter ) {
	
	if ( partition == "" || partition == iter->second.partitionName_ ) {
	  
	  FedDescriptions::range range = feds_.find( iter->second.partitionName_ );
	  if ( range == feds_.emptyRange() ) {

	    // Extract versions
	    deviceFactory(__func__)->setUsingStrips( usingStrips_ );
	    int16_t major = iter->second.fedVersion_.first; 
	    int16_t minor = iter->second.fedVersion_.second; 
	    if ( iter->second.fedVersion_.first == 0 && 
		 iter->second.fedVersion_.second == 0 ) {
	      major = -1; //@@ "current state" for fed factory!
	      minor = -1; //@@ "current state" for fed factory!
	    }

	    // Retrive FED descriptions
	    std::vector<FedDescription*> tmp1;
	    tmp1 = *( deviceFactory(__func__)->getFed9UDescriptions( iter->second.partitionName_, 
								     major, 
								     minor ) );
	    
	    // Make local copy 
	    std::vector<FedDescription*> tmp2;
	    Fed9U::Fed9UDeviceFactory::vectorCopy( tmp2, tmp1 );
	    
	    // Add to cache
	    feds_.loadNext( iter->second.partitionName_, tmp2 );

	    // Some debug
	    FedDescriptions::range feds = feds_.find( iter->second.partitionName_ );
	    std::stringstream ss;
	    ss << "[SiStripConfigDb::" << __func__ << "]"
	       << " Downloaded " << feds.size() 
	       << " FED descriptions to local cache for partition \""
	       << iter->second.partitionName_ << "\" and version " 
	       << iter->second.fedVersion_.first << "." 
	       << iter->second.fedVersion_.second << std::endl;
	    ss << "[SiStripConfigDb::" << __func__ << "]"
	       << " Cache holds connections for " 
	       << feds_.size() << " partitions.";
	    LogTrace(mlConfigDb_) << ss.str();
	    
	  }

	}

      }
	    
    } else {

#ifdef USING_NEW_DATABASE_MODEL

      std::vector<FedDescription*>* tmp1 = databaseCache(__func__)->getFed9UDescriptions();
      
      if ( tmp1 ) { 
	
	// Make local copy 
	std::vector<FedDescription*> tmp2;
	Fed9U::Fed9UDeviceFactory::vectorCopy( tmp2, *tmp1 );
	
	// Add to cache
	feds_.loadNext( "", tmp2 );
	
      } else {
	edm::LogWarning(mlConfigDb_)
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " NULL pointer to FED descriptions vector!";
      }

#endif

    }

  } catch (... ) { handleException( __func__ ); }
  
  // Create range object
  uint16_t np = 0;
  uint16_t nc = 0;
  FedDescriptions::range feds;
  if ( partition != "" ) { 
    feds = feds_.find( partition );
    np = 1;
    nc = feds.size();
  } else { 
    feds = FedDescriptions::range( feds_.find( dbParams_.partitions_.begin()->second.partitionName_ ).begin(),
				   feds_.find( dbParams_.partitions_.rbegin()->second.partitionName_ ).end() );
    np = feds_.size();
    nc = feds.size();
  }
  
  stringstream ss; 
  ss << "[SiStripConfigDb::" << __func__ << "]"
     << " Found " << nc << " FED descriptions";
  if ( !dbParams_.usingDb_ ) { ss << " in " << dbParams_.inputFedXmlFiles().size() << " 'fed.xml' file(s)"; }
  else { if ( !dbParams_.usingDbCache_ )  { ss << " in " << np << " database partition(s)"; } 
  else { ss << " from shared memory name '" << dbParams_.sharedMemory_ << "'"; } }
  if ( feds_.empty() ) { edm::LogWarning(mlConfigDb_) << ss.str(); }
  else { LogTrace(mlConfigDb_) << ss.str(); }
  
  return feds;
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::addFedDescriptions( std::string partition, std::vector<FedDescription*>& feds ) {

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

  SiStripDbParams::SiStripPartitions::iterator iter = dbParams_.partitions_.begin();
  SiStripDbParams::SiStripPartitions::iterator jter = dbParams_.partitions_.end();
  for ( ; iter != jter; ++iter ) { if ( partition == iter->second.partitionName_ ) { break; } }
  if ( iter == dbParams_.partitions_.end() ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Partition \"" << partition
       << "\" not found in partition list, "
       << " therefore cannot add FED descriptions!";
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
  FedDescriptions::range range = feds_.find( partition );
  if ( range == feds_.emptyRange() ) {
    
    // Make local copy 
    std::vector<FedDescription*> tmp;
#ifdef USING_NEW_DATABASE_MODEL
    Fed9U::Fed9UDeviceFactory::vectorCopy( tmp, feds );
#else
    tmp = feds;
#endif

    // Add to local cache
    feds_.loadNext( partition, tmp );
    
    // Some debug
    std::stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " Added " << feds.size() 
       << " FED descriptions to local cache for partition \""
       << iter->second.partitionName_ << "\" and version " 
       << iter->second.fedVersion_.first << "." 
       << iter->second.fedVersion_.second << std::endl;
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
  
  if ( dbParams_.usingDbCache_ ) {
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

    SiStripDbParams::SiStripPartitions::iterator iter = dbParams_.partitions_.begin();
    SiStripDbParams::SiStripPartitions::iterator jter = dbParams_.partitions_.end();
    for ( ; iter != jter; ++iter ) {
      
      if ( partition == "" || partition == iter->second.partitionName_ ) {
	
	FedDescriptions::range range = feds_.find( iter->second.partitionName_ );
	if ( range != feds_.emptyRange() ) {
	  
	  std::vector<FedDescription*> feds( range.begin(), range.end() );
// 	  std::vector<FedDescription*> feds; //@@ PATCH DUE TO BUG IN JONNY'S CODE
// 	  for ( FedDescriptions::data_iterator ii = range.begin(); ii != range.end(); ++ii ) {
// 	    if ( *ii ) { feds.push_back( (*ii)->clone() ); }
// 	  }
	  
	  for ( std::vector<FedDescription*>::iterator ii = feds.begin(); ii != feds.end(); ++ii ) {
	    if ( *ii ) {
	      std::stringstream sss;
	      sss << " BEFORE : "
		  << " ptr: " << *ii
		  << " SW id: " << (*ii)->getFedId()
		  << " HW id: " << (*ii)->getFedHardwareId();
	      LogTrace("TEST") << sss.str();
	    }
	  }
	  
	  deviceFactory(__func__)->setFed9UDescriptions( feds,
							 iter->second.partitionName_,
							 (uint16_t*)(&iter->second.fedVersion_.first), 
							 (uint16_t*)(&iter->second.fedVersion_.second),
							 1 ); // new major version

	  for ( std::vector<FedDescription*>::iterator ii = feds.begin(); ii != feds.end(); ++ii ) {
	    if ( *ii ) {
	      std::stringstream sss;
	      sss << " AFTER  : "
		  << " ptr: " << *ii
		  << " SW id: " << (*ii)->getFedId()
		  << " HW id: " << (*ii)->getFedHardwareId();
	      LogTrace("TEST") << sss.str();
	    }
	  }

	  // Some debug
	  std::stringstream ss;
	  ss << "[SiStripConfigDb::" << __func__ << "]"
	     << " Uploaded " << feds.size() 
	     << " FED descriptions to DB/xml for partition \""
	     << iter->second.partitionName_ << "\" and version " 
	     << iter->second.fedVersion_.first << "." 
	     << iter->second.fedVersion_.second << ".";
	  LogTrace(mlConfigDb_) << ss.str();
	  
	} else {
	  stringstream ss; 
	  ss << "[SiStripConfigDb::" << __func__ << "]" 
	     << " Vector of FED descriptions is empty for partition \"" 
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
  
  allowCalibUpload_ = true;
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::clearFedDescriptions( std::string partition ) {
  
  if ( feds_.empty() ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Found no cached FED descriptions!"; 
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
  // Reproduce temporary cache for "all partitions except specified one" (or clear all if none specified)
  FedDescriptions temporary_cache;
  if ( partition == ""  ) { temporary_cache = FedDescriptions(); }
  else {
    SiStripDbParams::SiStripPartitions::iterator iter = dbParams_.partitions_.begin();
    SiStripDbParams::SiStripPartitions::iterator jter = dbParams_.partitions_.end();
    for ( ; iter != jter; ++iter ) {
      if ( partition != iter->second.partitionName_ ) {
	FedDescriptions::range range = feds_.find( iter->second.partitionName_ );
	if ( range != feds_.emptyRange() ) {
	  temporary_cache.loadNext( partition, std::vector<FedDescription*>( range.begin(), range.end() ) );
	}
      } else {
	FedDescriptions::range range = feds_.find( iter->second.partitionName_ );
	if ( range != feds_.emptyRange() ) {
	  LogTrace(mlConfigDb_) 
	    << "[SiStripConfigDb::" << __func__ << "]"
	    << " Deleting FED descriptions for partition \""
	    << iter->second.partitionName_
	    << "\" from local cache...";
	}
      }
    }
  }
  
  // Delete objects in local cache for specified partition (or all if not specified) 
  FedDescriptions::range feds;
  if ( partition == "" ) { 
    feds = FedDescriptions::range( feds_.find( dbParams_.partitions_.begin()->second.partitionName_ ).begin(),
				   feds_.find( dbParams_.partitions_.rbegin()->second.partitionName_ ).end() );
  } else {
    SiStripDbParams::SiStripPartitions::iterator iter = dbParams_.partitions_.begin();
    SiStripDbParams::SiStripPartitions::iterator jter = dbParams_.partitions_.end();
    for ( ; iter != jter; ++iter ) { if ( partition == iter->second.partitionName_ ) { break; } }
    feds = feds_.find( iter->second.partitionName_ );
  }
  
  if ( feds != feds_.emptyRange() ) {

#ifdef USING_NEW_DATABASE_MODEL	
    std::vector<FedDescription*>::const_iterator ifed = feds.begin();
    std::vector<FedDescription*>::const_iterator jfed = feds.end();
    for ( ; ifed != jfed; ++ifed ) { if ( *ifed ) { delete *ifed; } }
#endif
    
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
      ss << "  Partition name   : " << iconn->first << std::endl;
      ss << "  Num of FED ids   : " << iconn->second.size() << std::endl;

      // Extract FED crates and ids
      std::map< uint16_t, vector<uint16_t> > feds;
      std::vector<FedDescription*>::const_iterator iter = iconn->second.begin();
      std::vector<FedDescription*>::const_iterator jter = iconn->second.end();
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
  
  if ( ( !dbParams_.usingDbCache_ && !deviceFactory(__func__) ) ||
       (  dbParams_.usingDbCache_ && !databaseCache(__func__) ) ) { 
    return std::make_pair( fedIds_.end(), fedIds_.end() );
  }
  
  try { 

    // Inhibit download of strip info
    bool using_strips = usingStrips_;
    if ( factory_ ) { factory_->setUsingStrips( false ); }
    FedDescriptions::range feds = getFedDescriptions( partition );
    if ( factory_ ) { factory_->setUsingStrips( using_strips ); }
    
    if ( !feds.empty() ) {
      std::vector<FedDescription*>::const_iterator ifed = feds.begin();
      std::vector<FedDescription*>::const_iterator jfed = feds.end();
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
  
  return std::make_pair( fedIds_.begin(), fedIds_.end() );

}

