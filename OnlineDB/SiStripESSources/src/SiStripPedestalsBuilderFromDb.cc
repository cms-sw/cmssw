// Last commit: $Id: SiStripPedestalsBuilderFromDb.cc,v 1.9 2008/06/06 08:05:12 bainbrid Exp $
// Latest tag:  $Name: V02-00-06 $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripESSources/src/SiStripPedestalsBuilderFromDb.cc,v $

#include "OnlineDB/SiStripESSources/interface/SiStripPedestalsBuilderFromDb.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
SiStripPedestalsBuilderFromDb::SiStripPedestalsBuilderFromDb( const edm::ParameterSet& pset ) 
  : SiStripPedestalsESSource( pset ) {
  LogTrace(mlESSources_) 
    << "[SiStripPedestalsBuilderFromDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
SiStripPedestalsBuilderFromDb::~SiStripPedestalsBuilderFromDb() {
  LogTrace(mlESSources_)
    << "[SiStripPedestalsBuilderFromDb::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
SiStripPedestals* SiStripPedestalsBuilderFromDb::makePedestals() {
  LogTrace(mlESSources_) 
    << "[SiStripPedestalsBuilderFromDb::" << __func__ << "]"
    << " Constructing Pedestals object...";
  
  // Create Pedestals object 
  SiStripPedestals* pedestals;
  condObjBuilder->getValue(pedestals);  
  return pedestals;
  
}

