// Last commit: $Id: DcuDetIdMap.cc,v 1.7 2006/10/10 14:35:45 bainbrid Exp $
// Latest tag:  $Name:  $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripConfigDb/src/DcuDetIdMap.cc,v $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::DcuDetIdMap& SiStripConfigDb::getDcuDetIdMap() {
  
  if ( !deviceFactory(__func__) ) { return dcuDetIdMap_; }
  if ( !resetDcuDetIdMap_ ) { return dcuDetIdMap_; }
  
  try {
    dcuDetIdMap_ = deviceFactory(__func__)->getInfos(); 
    resetDcuDetIdMap_ = false;
  }
  catch (... ) {
    handleException( __func__ );
  }
  
  // Debug
  stringstream ss; 
  ss << "[SiStripConfigDb::" << __func__ << "]";
  if ( devices_.empty() ) { ss << " Found no entries in DCU-DetId map"; }
  else { ss << " Found " << devices_.size() << " entries in DCU-DetId map"; }
  if ( !usingDb_ ) { ss << " in " << inputDcuInfoXml_.size() << " 'dcuinfo.xml' file(s)"; }
  else { ss << " in database partition '" << partition_.name_ << "'"; }
  if ( devices_.empty() ) { edm::LogWarning(mlConfigDb_) << ss; }
  else { LogTrace(mlConfigDb_) << ss; }

  return dcuDetIdMap_;
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::setDcuDetIdMap( const SiStripConfigDb::DcuDetIdMap& dcu_detid_map ) {
  resetDcuDetIdMap();
  dcuDetIdMap_ = dcu_detid_map;
  resetDcuDetIdMap_ = false;
}

// -----------------------------------------------------------------------------
//
void SiStripConfigDb::resetDcuDetIdMap() {
  dcuDetIdMap_.clear(); 
  resetDcuDetIdMap_ = true;
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::uploadDcuDetIdMap() {

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
