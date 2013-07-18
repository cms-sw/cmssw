// Last commit: $Id: SiStripPedestalsBuilderFromDb.cc,v 1.10 2008/07/17 10:27:59 giordano Exp $
// Latest tag:  $Name: V05-01-05 $
// Location:    $Source: /local/reps/CMSSW/CMSSW/OnlineDB/SiStripESSources/src/SiStripPedestalsBuilderFromDb.cc,v $

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

