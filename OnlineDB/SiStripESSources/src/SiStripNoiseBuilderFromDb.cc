// Last commit: $Id: SiStripNoiseBuilderFromDb.cc,v 1.11 2008/07/17 10:27:59 giordano Exp $
// Latest tag:  $Name: V05-01-05 $
// Location:    $Source: /local/reps/CMSSW/CMSSW/OnlineDB/SiStripESSources/src/SiStripNoiseBuilderFromDb.cc,v $

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

