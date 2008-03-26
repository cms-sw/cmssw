// Last commit: $Id: DcuDetIdMap.cc,v 1.11 2008/02/06 17:13:12 bainbrid Exp $
// Latest tag:  $Name:  $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripConfigDb/src/DcuDetIdMap.cc,v $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::DcuDetIdMap& SiStripConfigDb::getDcuDetIdMap() {

  dcuDetIdMap_.clear();
  
  if ( ( !dbParams_.usingDbCache_ && !deviceFactory(__func__) ) ||
       (  dbParams_.usingDbCache_ && !databaseCache(__func__) ) ) { return dcuDetIdMap_; }
  
  try {

    if ( !dbParams_.usingDbCache_ ) { 

      dcuDetIdMap_ = deviceFactory(__func__)->getInfos(); 

    } else {

#ifdef USING_DATABASE_CACHE
      DcuDetIdMap* tmp = 0; //@@ databaseCache(__func__)->getDcuDetIdMap();
      if ( tmp ) { dcuDetIdMap_ = *tmp; } //@@ bug here?
      else {
	edm::LogWarning(mlConfigDb_)
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " NULL pointer to Dcu-DetId map!"
	  << " NOT IMPLEMENTED YET!!!"; //@@
      }
#endif
      
    }
    
  } catch (... ) { handleException( __func__ ); }
  
  stringstream ss; 
  ss << "[SiStripConfigDb::" << __func__ << "]" 
     << " Found " << dcuDetIdMap_.size() 
     << " entries in DCU-DetId map"; 
  if ( !dbParams_.usingDb_ ) { ss << " in " << dbParams_.inputDcuInfoXml_.size() << " 'dcuinfo.xml' file(s)"; }
  else { if ( !dbParams_.usingDbCache_ )  { ss << " in database partition '" << dbParams_.partition_ << "'"; } 
  else { ss << " from shared memory name '" << dbParams_.sharedMemory_ << "'"; } }
  if ( devices_.empty() ) { edm::LogWarning(mlConfigDb_) << ss.str(); }
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
