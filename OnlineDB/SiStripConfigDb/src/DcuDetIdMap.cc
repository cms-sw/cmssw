// Last commit: $Id: DcuDetIdMap.cc,v 1.10 2007/11/20 22:39:27 bainbrid Exp $
// Latest tag:  $Name:  $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripConfigDb/src/DcuDetIdMap.cc,v $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::DcuDetIdMap& SiStripConfigDb::getDcuDetIdMap() {
  
  if ( !deviceFactory(__func__) ) { return dcuDetIdMap_; }
  
  try {
    dcuDetIdMap_ = deviceFactory(__func__)->getInfos(); 
  }
  catch (... ) {
    handleException( __func__ );
  }
  
  // Debug
  stringstream ss; 
  ss << "[SiStripConfigDb::" << __func__ << "]";
  if ( devices_.empty() ) { ss << " Found no entries in DCU-DetId map"; }
  else { ss << " Found " << devices_.size() << " entries in DCU-DetId map"; }
  if ( !dbParams_.usingDb_ ) { ss << " in " << dbParams_.inputDcuInfoXml_.size() << " 'dcuinfo.xml' file(s)"; }
  else { ss << " in database partition '" << dbParams_.partition_ << "'"; }
  if ( devices_.empty() ) { edm::LogWarning(mlConfigDb_) << ss; }
  else { LogTrace(mlConfigDb_) << ss; }

  return dcuDetIdMap_;
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
