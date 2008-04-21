// Last commit: $Id: DcuDetIdMap.cc,v 1.15 2008/04/11 13:27:33 bainbrid Exp $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::DcuDetIdMap& SiStripConfigDb::getDcuDetIdMap() {
  
//     // DCU-DetId 
//     try { 
// #ifdef USING_NEW_DATABASE_MODEL
//       deviceFactory(__func__)->addDetIdPartition( ip->second.partitionName_,
// 						  ip->second.dcuVersion_.first, 
// 						  ip->second.dcuVersion_.second );
// #else
//       deviceFactory(__func__)->addDetIdPartition( ip->second.partitionName_ );
// #endif
//     } catch (...) { 
//       std::stringstream ss;
//       ss << "Attempted to 'addDetIdPartition' for partition: " << ip->second.partitionName_;
//       handleException( __func__, ss.str() );
//     }

  dcuDetIdMap_.clear();
  
  if ( ( !dbParams_.usingDbCache_ && !deviceFactory(__func__) ) ||
       (  dbParams_.usingDbCache_ && !databaseCache(__func__) ) ) { return dcuDetIdMap_; }
  
  try {

    if ( !dbParams_.usingDbCache_ ) { 
      
      dcuDetIdMap_ = deviceFactory(__func__)->getInfos(); 
      
    } else {
      
#ifdef USING_NEW_DATABASE_MODEL
      DcuDetIdMap* tmp = databaseCache(__func__)->getInfos();
      if ( tmp ) { dcuDetIdMap_ = *tmp; } 
      else {
	edm::LogWarning(mlConfigDb_)
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " NULL pointer to Dcu-DetId map!";
      }
#endif
      
    }
    
  } catch (... ) { handleException( __func__ ); }
  
  stringstream ss; 
  ss << "[SiStripConfigDb::" << __func__ << "]" 
     << " Found " << dcuDetIdMap_.size() 
     << " entries in DCU-DetId map"; 
  if ( !dbParams_.usingDb_ ) { ss << " in " << dbParams_.partitions_.begin()->second.inputDcuInfoXml_.size() << " 'dcuinfo.xml' file(s)"; }
  else { if ( !dbParams_.usingDbCache_ )  { ss << " in database partition '" << dbParams_.partitions_.begin()->second.partitionName_ << "'"; } 
  else { ss << " from shared memory name '" << dbParams_.sharedMemory_ << "'"; } }
  if ( dcuDetIdMap_.empty() ) { edm::LogWarning(mlConfigDb_) << ss.str(); }
  else { LogTrace(mlConfigDb_) << ss.str(); }

  return dcuDetIdMap_;
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::uploadDcuDetIdMap() {

  if ( dbParams_.usingDbCache_ ) {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]" 
      << " Using database cache! No uploads allowed!"; 
    return;
  }

  if ( !deviceFactory(__func__) ) { return; }

  try {
    deviceFactory(__func__)->deleteHashMapTkDcuInfo();
    deviceFactory(__func__)->setTkDcuInfo( dcuDetIdMap_ );
    deviceFactory(__func__)->addAllDetId();
  }
  catch (... ) {
    handleException( __func__, "Problems updating objects in TkDcuInfoFactory!" );
  }
  
}
