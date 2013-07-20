// Last commit: $Id: SiStripGainBuilderFromDb.cc,v 1.2 2013/05/30 21:52:09 gartung Exp $
// Latest tag:  $Name: CMSSW_6_2_0 $
// Location:    $Source: /local/reps/CMSSW/CMSSW/OnlineDB/SiStripESSources/src/SiStripGainBuilderFromDb.cc,v $

#include "OnlineDB/SiStripESSources/interface/SiStripGainBuilderFromDb.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "OnlineDB/SiStripESSources/interface/SiStripCondObjBuilderFromDb.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <vector>
#include <string>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
SiStripGainBuilderFromDb::SiStripGainBuilderFromDb( const edm::ParameterSet& pset ) 
  : SiStripGainESSource( pset )
{
  LogTrace(mlESSources_) 
    << "[SiStripGainBuilderFromDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
SiStripGainBuilderFromDb::~SiStripGainBuilderFromDb() {
  LogTrace(mlESSources_)
    << "[SiStripGainBuilderFromDb::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
SiStripApvGain* SiStripGainBuilderFromDb::makeGain() {
  LogTrace(mlESSources_) 
    << "[SiStripGainBuilderFromDb::" << __func__ << "]"
    << " Constructing Gain object...";

  /** Service to access onlineDB and extract pedestal/gain */
  edm::Service<SiStripCondObjBuilderFromDb> condObjBuilder_;
  
  // Create Gain object 
  SiStripApvGain* gain;
  condObjBuilder_->getValue(gain);
  return gain;
  
}

