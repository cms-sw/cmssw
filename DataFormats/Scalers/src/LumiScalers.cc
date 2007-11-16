
/*
 *   File: DataFormats/Scalers/src/LumiScalers.cc   (W.Badgett)
 */

#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/Scalers/interface/ScalersRaw.h"

LumiScalers::LumiScalers() : 
   version_(0),
   normalization_(0),
   instantLumi_(0),
   instantLumiErr_(0),
   instantLumiQlty_(0),
   instantETLumi_(0),
   instantETLumiErr_(0),
   instantETLumiQlty_(0),
   instantOccLumi_(nOcc),
   instantOccLumiErr_(nOcc),
   instantOccLumiQlty_(nOcc),
   lumiNoise_(nOcc),
   sectionNumber_(0),
   startOrbit_(0),
   numOrbits_(0)
{ 
}

LumiScalers::LumiScalers(const unsigned char * rawData)
{ 
  LumiScalers();
  version_ = ((int *)rawData)[0];
  if ( version_ == 1 )
  {
    ScalersEventRecordRaw_v1 * raw 
      = (struct ScalersEventRecordRaw_v1 *)rawData;
    normalization_     = raw->lumi.Normalization;
    instantLumi_       = raw->lumi.InstantLumi;
    instantLumiErr_    = raw->lumi.InstantLumiErr;
    instantLumiQlty_   = raw->lumi.InstantLumiQlty;
    instantETLumi_     = raw->lumi.InstantETLumi;
    instantETLumiErr_  = raw->lumi.InstantETLumiErr;
    instantETLumiQlty_ = raw->lumi.InstantETLumiQlty;
    for ( int i=0; i<ScalersRaw::N_LUMI_OCC_v1; i++)
    {
      instantOccLumi_[i]     = raw->lumi.InstantOccLumi[i];
      instantOccLumiErr_[i]  = raw->lumi.InstantOccLumiErr[i];
      instantOccLumiQlty_[i] = raw->lumi.InstantOccLumiQlty[i];
      lumiNoise_[i]          = raw->lumi.lumiNoise[i];
    }
    sectionNumber_ = raw->lumi.sectionNumber;
    startOrbit_    = raw->lumi.startOrbit;
    numOrbits_     = raw->lumi.numOrbits;
  }
}

LumiScalers::~LumiScalers() { } 


/// Pretty-print operator for LumiScalers
std::ostream& operator<<(std::ostream& s, const LumiScalers& c) 
{
  s << " LumiScalers: ";
  return s;
}
