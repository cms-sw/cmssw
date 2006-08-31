// Last commit: $Id: DcuDetIdMap.cc,v 1.5 2006/07/26 11:27:19 bainbrid Exp $
// Latest tag:  $Name: V00-01-02 $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripConfigDb/src/DcuDetIdMap.cc,v $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"

using namespace std;

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::DcuDetIdMap& SiStripConfigDb::getDcuDetIdMap() {
  edm::LogInfo(logCategory_) << "[" << __PRETTY_FUNCTION__ << "]"
			     << " Retrieving DetId-DCU mapping...";
  
  if ( !deviceFactory(__FUNCTION__) ) { return dcuDetIdMap_; }
  if ( !resetDcuDetIdMap_ ) { return dcuDetIdMap_; }
  
  try {
    dcuDetIdMap_ = deviceFactory(__FUNCTION__)->getInfos(); 
    resetDcuDetIdMap_ = false;
  }
  catch (... ) {
    handleException( __FUNCTION__ );
  }
  
  stringstream ss; 
  if ( dcuDetIdMap_.empty() ) {
    ss << "[" << __PRETTY_FUNCTION__ << "]"
       << " No DCU-DetId map found";
    if ( !usingDb_ ) { ss << " in input 'dcuinfo.xml' file " << inputDcuInfoXml_; }
    else { ss << " in database partition '" << partition_.name_ << "'"; }
    edm::LogWarning(logCategory_) << ss.str();
  } else {
    ss << "[" << __PRETTY_FUNCTION__ << "]"
       << " Found " << dcuDetIdMap_.size() << " entries in DCU-DetId map";
    if ( !usingDb_ ) { ss << " in input 'module.xml' file " << inputDcuInfoXml_; }
    else { ss << " in database partition '" << partition_.name_ << "'"; }
    edm::LogInfo(logCategory_) << ss.str();
  }

  return dcuDetIdMap_;
}
// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::setDcuDetIdMap( const SiStripConfigDb::DcuDetIdMap& dcu_detid_map ) {
  edm::LogInfo(logCategory_) << "[" << __PRETTY_FUNCTION__ << "]"
			     << " Setting DetId-DCU mapping...";
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

  if ( !deviceFactory(__FUNCTION__) ) { return; }

  try {
    deviceFactory(__FUNCTION__)->deleteHashMapTkDcuInfo();
    deviceFactory(__FUNCTION__)->setTkDcuInfo( dcuDetIdMap_ );
    deviceFactory(__FUNCTION__)->addAllDetId();
  }
  catch (... ) {
    handleException( __FUNCTION__, "Problems updating objects in TkDcuInfoFactory!" );
  }
}
