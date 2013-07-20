// Last commit: $Id: AnalysisDescriptions.cc,v 1.13 2009/04/06 16:57:28 lowette Exp $
// Latest tag:  $Name: CMSSW_6_2_0 $
// Location:    $Source: /local/reps/CMSSW/CMSSW/OnlineDB/SiStripConfigDb/src/AnalysisDescriptions.cc,v $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** T_UNKNOWN,
    T_ANALYSIS_FASTFEDCABLING,
    T_ANALYSIS_TIMING,
    T_ANALYSIS_OPTOSCAN,
    T_ANALYSIS_VPSPSCAN,
    T_ANALYSIS_PEDESTAL,
    T_ANALYSIS_APVLATENCY,
    T_ANALYSIS_FINEDELAY,
    T_ANALYSIS_CALIBRATION.
*/
SiStripConfigDb::AnalysisDescriptionsRange SiStripConfigDb::getAnalysisDescriptions( AnalysisType analysis_type,
										     std::string partition ) {
  
  // Check
  if ( ( !dbParams_.usingDbCache() && !deviceFactory(__func__) ) ||
       (  dbParams_.usingDbCache() && !databaseCache(__func__) ) ) { 
    return analyses_.emptyRange();
  }
  
  try { 

    if ( !dbParams_.usingDbCache() ) { 
      
      SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partitions().begin();
      SiStripDbParams::SiStripPartitions::const_iterator jter = dbParams_.partitions().end();
      for ( ; iter != jter; ++iter ) {
	
	if ( partition == "" || partition == iter->second.partitionName() ) {
	  
	  if ( iter->second.partitionName() == SiStripPartition::defaultPartitionName_ ) { continue; }

	  AnalysisDescriptionsRange range = analyses_.find( iter->second.partitionName() );
	  if ( range == analyses_.emptyRange() ) {
    
	    AnalysisDescriptionsV tmp1;
	    if ( analysis_type == AnalysisDescription::T_ANALYSIS_FASTFEDCABLING ) {
	      tmp1 = deviceFactory(__func__)->getAnalysisHistory( iter->second.partitionName(), 
								  iter->second.fastCablingVersion().first,
								  iter->second.fastCablingVersion().second,
								  analysis_type );
	    } else if ( analysis_type == AnalysisDescription::T_ANALYSIS_TIMING ) {
	      tmp1 = deviceFactory(__func__)->getAnalysisHistory( iter->second.partitionName(), 
								  iter->second.apvTimingVersion().first,
								  iter->second.apvTimingVersion().second,
								  analysis_type );
	    } else if ( analysis_type == AnalysisDescription::T_ANALYSIS_OPTOSCAN ) {
	      tmp1 = deviceFactory(__func__)->getAnalysisHistory( iter->second.partitionName(), 
								  iter->second.optoScanVersion().first,
								  iter->second.optoScanVersion().second,
								  analysis_type );
	    } else if ( analysis_type == AnalysisDescription::T_ANALYSIS_VPSPSCAN ) {
	      tmp1 = deviceFactory(__func__)->getAnalysisHistory( iter->second.partitionName(), 
								  iter->second.vpspScanVersion().first,
								  iter->second.vpspScanVersion().second,
								  analysis_type );
	    } else if ( analysis_type == AnalysisDescription::T_ANALYSIS_CALIBRATION ) {
	      tmp1 = deviceFactory(__func__)->getAnalysisHistory( iter->second.partitionName(), 
								  iter->second.apvCalibVersion().first,
								  iter->second.apvCalibVersion().second,
								  analysis_type );
	    } else if ( analysis_type == AnalysisDescription::T_ANALYSIS_PEDESTALS ) { 
	      tmp1 = deviceFactory(__func__)->getAnalysisHistory( iter->second.partitionName(), 
								  iter->second.pedestalsVersion().first,
								  iter->second.pedestalsVersion().second,
								  analysis_type );
	    } else if ( analysis_type == AnalysisDescription::T_ANALYSIS_APVLATENCY ) {
	      tmp1 = deviceFactory(__func__)->getAnalysisHistory( iter->second.partitionName(), 
								  iter->second.apvLatencyVersion().first,
								  iter->second.apvLatencyVersion().second,
								  analysis_type );
	    } else if ( analysis_type == AnalysisDescription::T_ANALYSIS_FINEDELAY ) {
	      tmp1 = deviceFactory(__func__)->getAnalysisHistory( iter->second.partitionName(), 
								  iter->second.fineDelayVersion().first,
								  iter->second.fineDelayVersion().second,
								  analysis_type );
	    } else {
	      std::stringstream ss;
	      ss << "[SiStripConfigDb::" << __func__ << "]"
		 << " Unexpected analysis type \"" 
		 << analysisType( analysis_type ) 
		 << "\"! Aborting download...";
	      edm::LogWarning(mlConfigDb_) << ss.str();
	      return analyses_.emptyRange();
	    }
	    
	    // Make local copy 
	    AnalysisDescriptionsV tmp2;
	    CommissioningAnalysisFactory::vectorCopy( tmp1, tmp2 );
	    
	    // Add to cache
	    analyses_.loadNext( iter->second.partitionName(), tmp2 );
	    
 	    // Some debug
	    AnalysisDescriptionsRange anals = analyses_.find( iter->second.partitionName() );
	    std::stringstream ss;
	    ss << "[SiStripConfigDb::" << __func__ << "]"
	       << " Downloaded " << anals.size() 
	       << " analysis descriptions of type \""
	       << analysisType( analysis_type )
	       << "\" to local cache for partition \""
	       << iter->second.partitionName() << "\"" << std::endl;
	    ss << "[SiStripConfigDb::" << __func__ << "]"
	       << " Cache holds analysis descriptions for " 
	       << analyses_.size() << " partitions.";
	    LogTrace(mlConfigDb_) << ss.str();

	  }
	}
      }

    } else { // Use database cache
      std::stringstream ss;
      ss << "[SiStripConfigDb::" << __func__ << "]"
	 << " No database cache for analysis objects!";
      edm::LogWarning(mlConfigDb_) << ss.str();
    }
    
  } catch (...) { handleException( __func__ ); }
  
  // Create range object
  uint16_t np = 0;
  uint16_t nc = 0;
  AnalysisDescriptionsRange anals = analyses_.emptyRange();
  if ( partition != "" ) { 
    anals = analyses_.find( partition );
    np = 1;
    nc = anals.size();
  } else { 
    if ( !analyses_.empty() ) {
      anals = AnalysisDescriptionsRange( analyses_.find( dbParams_.partitions().begin()->second.partitionName() ).begin(),
					 analyses_.find( (--(dbParams_.partitions().end()))->second.partitionName() ).end() );
    } else { anals = analyses_.emptyRange(); }
    np = analyses_.size();
    nc = anals.size();
  }
  
  stringstream ss; 
  ss << "[SiStripConfigDb::" << __func__ << "]"
     << " Found " << nc << " analysis descriptions";
  if ( !dbParams_.usingDbCache() )  { ss << " in " << np << " database partition(s)"; } 
  else { ss << " from shared memory name '" << dbParams_.sharedMemory() << "'"; } 
  if ( analyses_.empty() ) { edm::LogWarning(mlConfigDb_) << ss.str(); }
  else { LogTrace(mlConfigDb_) << ss.str(); }
  
  return anals;
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::addAnalysisDescriptions( std::string partition, AnalysisDescriptionsV& anals ) {

  if ( !deviceFactory(__func__) ) { return; }

  if ( partition.empty() ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Partition string is empty,"
       << " therefore cannot add analysis descriptions to local cache!"; 
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
  if ( anals.empty() ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Vector of analysis descriptions is empty,"
       << " therefore cannot add analysis descriptions to local cache!"; 
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
       << " therefore cannot add analysis descriptions!";
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
  AnalysisDescriptionsRange range = analyses_.find( partition );
  if ( range == analyses_.emptyRange() ) {
    
    // Make local copy 
    AnalysisDescriptionsV tmp;
    CommissioningAnalysisFactory::vectorCopy( anals, tmp );
    
    // Add to local cache
    analyses_.loadNext( partition, tmp );

    // Some debug
    std::stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " Added " << anals.size() 
       << " analysis descriptions to local cache for partition \""
       << partition << "\"."
       << " (Cache holds analysis descriptions for " 
       << analyses_.size() << " partitions.)";
    LogTrace(mlConfigDb_) << ss.str();
    
  } else {
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Partition \"" << partition
       << "\" already found in local cache, "
       << " therefore cannot add analysis descriptions!";
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::uploadAnalysisDescriptions( bool calibration_for_physics,
						  std::string partition ) {

  if ( dbParams_.usingDbCache() ) {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]" 
      << " Using database cache! No uploads allowed!"; 
    return;
  }
  
  if ( !deviceFactory(__func__) ) { return; }

  if ( analyses_.empty() ) { 
    edm::LogWarning(mlConfigDb_) 
      << "[SiStripConfigDb::" << __func__ << "]" 
      << " Found no cached analysis descriptions, therefore no upload!"; 
    return; 
  }
  
  if ( calibration_for_physics && !allowCalibUpload_ ) {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Attempting to upload calibration constants"
      << " without uploading any hardware descriptions!"
      << " Aborting upload...";
    return;
  } else { allowCalibUpload_ = false; }
  
  try { 

    SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partitions().begin();
    SiStripDbParams::SiStripPartitions::const_iterator jter = dbParams_.partitions().end();
    for ( ; iter != jter; ++iter ) {
      
      if ( partition == "" || partition == iter->second.partitionName() ) {
	
	AnalysisDescriptionsRange range = analyses_.find( iter->second.partitionName() );
	if ( range != analyses_.emptyRange() ) {
	  
	  AnalysisDescriptionsV anals( range.begin(), range.end() );
	  
	  AnalysisType analysis_type = AnalysisDescription::T_UNKNOWN;
	  if ( anals.front() ) { analysis_type = anals.front()->getType(); }
	  if ( analysis_type == AnalysisDescription::T_UNKNOWN ) {
	    edm::LogWarning(mlConfigDb_)
	      << "[SiStripConfigDb::" << __func__ << "]"
	      << " Analysis type is UNKNOWN. Aborting upload!";
	    return;
	  }
	  
	  uint32_t version = deviceFactory(__func__)->uploadAnalysis( iter->second.runNumber(), 
								      iter->second.partitionName(), 
								      analysis_type,
								      anals,
								      calibration_for_physics );

	  // Update current state with analysis descriptions
	  if ( calibration_for_physics ) { deviceFactory(__func__)->uploadAnalysisState( version ); }
	  
	  // Some debug
	  std::stringstream ss;
	  ss << "[SiStripConfigDb::" << __func__ << "]"
	     << " Uploaded " << anals.size() 
	     << " device descriptions to database for partition \""
	     << iter->second.partitionName() << "\".";
	  LogTrace(mlConfigDb_) << ss.str();
	  
	} else {
	  stringstream ss; 
	  ss << "[SiStripConfigDb::" << __func__ << "]" 
	     << " Vector of device descriptions is empty for partition \"" 
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
void SiStripConfigDb::clearAnalysisDescriptions( std::string partition ) {
  LogTrace(mlConfigDb_) << "[SiStripConfigDb::" << __func__ << "]";
  
  if ( analyses_.empty() ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Found no cached analysis descriptions!"; 
    //edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
  // Reproduce temporary cache for "all partitions except specified one" (or clear all if none specified)
  AnalysisDescriptions temporary_cache;
  if ( partition == ""  ) { temporary_cache = AnalysisDescriptions(); }
  else {
    SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partitions().begin();
    SiStripDbParams::SiStripPartitions::const_iterator jter = dbParams_.partitions().end();
    for ( ; iter != jter; ++iter ) {
      if ( partition != iter->second.partitionName() ) {
	AnalysisDescriptionsRange range = analyses_.find( iter->second.partitionName() );
	if ( range != analyses_.emptyRange() ) {
	  temporary_cache.loadNext( partition, AnalysisDescriptionsV( range.begin(), range.end() ) );
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
  AnalysisDescriptionsRange anals = analyses_.emptyRange();
  if ( partition == "" ) { 
    if ( !analyses_.empty() ) {
      anals = AnalysisDescriptionsRange( analyses_.find( dbParams_.partitions().begin()->second.partitionName() ).begin(),
					 analyses_.find( (--(dbParams_.partitions().end()))->second.partitionName() ).end() );
    } else { anals = analyses_.emptyRange(); }
  } else {
    SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partitions().begin();
    SiStripDbParams::SiStripPartitions::const_iterator jter = dbParams_.partitions().end();
    for ( ; iter != jter; ++iter ) { if ( partition == iter->second.partitionName() ) { break; } }
    anals = analyses_.find( iter->second.partitionName() );
  }
  
  if ( anals != analyses_.emptyRange() ) {
    AnalysisDescriptionsV::const_iterator ianal = anals.begin();
    AnalysisDescriptionsV::const_iterator janal = anals.end();
    for ( ; ianal != janal; ++ianal ) { if ( *ianal ) { delete *ianal; } }
  } else {
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]";
    if ( partition == "" ) { ss << " Found no analysis descriptions in local cache!"; }
    else { ss << " Found no analysis descriptions in local cache for partition \"" << partition << "\"!"; }
    edm::LogWarning(mlConfigDb_) << ss.str(); 
  }
  
  // Overwrite local cache with temporary cache
  analyses_ = temporary_cache; 

}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::printAnalysisDescriptions( std::string partition ) {
  
  std::stringstream ss;
  ss << "[SiStripConfigDb::" << __func__ << "]"
     << " Contents of AnalysisDescriptions container:" << std::endl;
  ss << " Number of partitions: " << analyses_.size() << std::endl;
  
  // Loop through partitions
  uint16_t cntr = 0;
  AnalysisDescriptions::const_iterator ianal = analyses_.begin();
  AnalysisDescriptions::const_iterator janal = analyses_.end();
  for ( ; ianal != janal; ++ianal ) {

    cntr++;
    if ( partition == "" || partition == ianal->first ) {
      
      ss << "  Partition number : " << cntr << " (out of " << analyses_.size() << ")" << std::endl;
      ss << "  Partition name   : \"" << ianal->first << "\"" << std::endl;
      ss << "  Num of analyses  : " << ianal->second.size() << std::endl;
      
      // Extract FEC crate, slot, etc
      std::map< uint32_t, vector<uint32_t> > analyses;
      AnalysisDescriptionsV::const_iterator iter = ianal->second.begin();
      AnalysisDescriptionsV::const_iterator jter = ianal->second.end();
      for ( ; iter != jter; ++iter ) { 
	if ( *iter ) { 
	  DeviceAddress addr = deviceAddress( **iter );
	  uint32_t key  = SiStripFecKey( addr.fecCrate_, 
					 addr.fecSlot_, 
					 addr.fecRing_, 
					 0, 
					 0, 
					 0, 
					 0 ).key();
	  uint32_t data = SiStripFecKey( addr.fecCrate_, 
					 addr.fecSlot_, 
					 addr.fecRing_, 
					 addr.ccuAddr_, 
					 addr.ccuChan_, 
					 addr.lldChan_, 
					 addr.i2cAddr_ ).key();
	  if ( find( analyses[key].begin(), analyses[key].end(), data ) == analyses[key].end() ) { 
	    analyses[key].push_back( data );
	  }
	}
      }
      
      // Sort contents
      std::map< uint32_t, std::vector<uint32_t> > tmp;
      std::map< uint32_t, std::vector<uint32_t> >::const_iterator ii = analyses.begin();
      std::map< uint32_t, std::vector<uint32_t> >::const_iterator jj = analyses.end();
      for ( ; ii != jj; ++ii ) {
	std::vector<uint32_t> temp = ii->second;
	std::sort( temp.begin(), temp.end() );
	std::vector<uint32_t>::const_iterator iii = temp.begin();
	std::vector<uint32_t>::const_iterator jjj = temp.end();
	for ( ; iii != jjj; ++iii ) { tmp[ii->first].push_back( *iii ); }
      }
      analyses.clear();
      analyses = tmp;
      
      // Print FEC crate, slot, etc...
      std::map< uint32_t, std::vector<uint32_t> >::const_iterator ianal = analyses.begin();
      std::map< uint32_t, std::vector<uint32_t> >::const_iterator janal = analyses.end();
      for ( ; ianal != janal; ++ianal ) {
	SiStripFecKey key(ianal->first);
	ss << "  Found " << std::setw(3) << ianal->second.size()
	   << " analyses for FEC crate/slot/ring " 
	   << key.fecCrate() << "/"
	   << key.fecSlot() << "/"
	   << key.fecRing();
	//<< " (ccu/module/lld/i2c): ";
	// 	if ( !ianal->second.empty() ) { 
	// 	  uint16_t first = ianal->second.front();
	// 	  uint16_t last = ianal->second.front();
	// 	  std::vector<uint32_t>::const_iterator chan = ianal->second.begin();
	// 	  for ( ; chan != ianal->second.end(); chan++ ) { 
	// 	    if ( chan != ianal->second.begin() ) {
	// 	      if ( *chan != last+1 ) { 
	// 		ss << std::setw(2) << first << "->" << std::setw(2) << last << ", ";
	// 		if ( chan != ianal->second.end() ) { first = *(chan+1); }
	// 	      } 
	// 	    }
	// 	    last = *chan;
	// 	  }
	// 	  if ( first != last ) { ss << std::setw(2) << first << "->" << std::setw(2) << last; }
	ss << std::endl;
      }

    }
    
  }
  
  LogTrace(mlConfigDb_) << ss.str();

}

// -----------------------------------------------------------------------------
// 
SiStripConfigDb::DeviceAddress SiStripConfigDb::deviceAddress( const AnalysisDescription& desc ) {
  
  DeviceAddress addr;
  try {
    addr.fecCrate_ = static_cast<uint16_t>( desc.getCrate() + sistrip::FEC_CRATE_OFFSET );
    addr.fecSlot_  = static_cast<uint16_t>( desc.getSlot() );
    addr.fecRing_  = static_cast<uint16_t>( desc.getRing() + sistrip::FEC_RING_OFFSET );
    addr.ccuAddr_  = static_cast<uint16_t>( desc.getCcuAdr() );
    addr.ccuChan_  = static_cast<uint16_t>( desc.getCcuChan() );
    addr.lldChan_  = static_cast<uint16_t>( SiStripFecKey::lldChan( desc.getI2cAddr() ) );
    addr.i2cAddr_  = static_cast<uint16_t>( desc.getI2cAddr() );
    addr.fedId_    = static_cast<uint16_t>( desc.getFedId() ); //@@ offset required? crate/slot needed?
    addr.feUnit_   = static_cast<uint16_t>( desc.getFeUnit() );
    addr.feChan_   = static_cast<uint16_t>( desc.getFeChan() );
  } catch (...) { handleException( __func__ ); }
  
  return addr;
}

// -----------------------------------------------------------------------------
//
std::string SiStripConfigDb::analysisType( AnalysisType analysis_type ) const {
  if      ( analysis_type == AnalysisDescription::T_ANALYSIS_FASTFEDCABLING ) { return "FAST_CABLING"; }
  else if ( analysis_type == AnalysisDescription::T_ANALYSIS_TIMING )         { return "APV_TIMING"; }
  else if ( analysis_type == AnalysisDescription::T_ANALYSIS_OPTOSCAN )       { return "OPTO_SCAN"; }
  else if ( analysis_type == AnalysisDescription::T_ANALYSIS_PEDESTALS )      { return "PEDESTALS"; }
  else if ( analysis_type == AnalysisDescription::T_ANALYSIS_APVLATENCY )     { return "APV_LATENCY"; }
  else if ( analysis_type == AnalysisDescription::T_ANALYSIS_FINEDELAY )      { return "FINE_DELAY"; }
  else if ( analysis_type == AnalysisDescription::T_ANALYSIS_CALIBRATION )    { return "CALIBRATION"; }
  else if ( analysis_type == AnalysisDescription::T_UNKNOWN )                 { return "UNKNOWN ANALYSIS TYPE"; }
  else { return "UNDEFINED ANALYSIS TYPE"; }
}
