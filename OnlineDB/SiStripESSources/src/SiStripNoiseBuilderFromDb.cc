// Last commit: $Id: SiStripNoiseBuilderFromDb.cc,v 1.10 2008/06/06 08:05:12 bainbrid Exp $
// Latest tag:  $Name: V02-00-06 $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripESSources/src/SiStripNoiseBuilderFromDb.cc,v $

#include "OnlineDB/SiStripESSources/interface/SiStripNoiseBuilderFromDb.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
SiStripNoiseBuilderFromDb::SiStripNoiseBuilderFromDb( const edm::ParameterSet& pset ) 
  : SiStripNoiseESSource( pset )
{
  LogTrace(mlESSources_) 
    << "[SiStripNoiseBuilderFromDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
SiStripNoiseBuilderFromDb::~SiStripNoiseBuilderFromDb() {
  LogTrace(mlESSources_)
    << "[SiStripNoiseBuilderFromDb::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
SiStripNoises* SiStripNoiseBuilderFromDb::makeNoise() {
  LogTrace(mlESSources_) 
    << "[SiStripNoiseBuilderFromDb::" << __func__ << "]"
    << " Constructing Noise object...";
  
  // Create Noise object 
  SiStripNoises* noise;
  condObjBuilder->getValue(noise);
  return noise;
  
}

