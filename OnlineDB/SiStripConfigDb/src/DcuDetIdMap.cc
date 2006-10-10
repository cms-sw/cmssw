// Last commit: $Id: DcuDetIdMap.cc,v 1.6 2006/08/31 19:49:41 bainbrid Exp $
// Latest tag:  $Name:  $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripConfigDb/src/DcuDetIdMap.cc,v $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::DcuDetIdMap& SiStripConfigDb::getDcuDetIdMap() {
  edm::LogInfo(mlConfigDb_) << __func__ << " Retrieving DetId-DCU mapping...";
  
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
  ostringstream os; 
  if ( devices_.empty() ) { os << " Found no entries in DCU-DetId map"; }
  else { os << " Found " << devices_.size() << " entries in DCU-DetId map"; }
  if ( !usingDb_ ) { os << " in " << inputDcuInfoXml_.size() << " 'dcuinfo.xml' file(s)"; }
  else { os << " in database partition '" << partition_.name_ << "'"; }
  if ( devices_.empty() ) { edm::LogError(mlConfigDb_) << os; }
  else { LogTrace(mlConfigDb_) << os; }

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
