
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
      instantOccLumi_.push_back(raw->lumi.InstantOccLumi[i]);
      instantOccLumiErr_.push_back(raw->lumi.InstantOccLumiErr[i]);
      instantOccLumiQlty_.push_back(raw->lumi.InstantOccLumiQlty[i]);
      lumiNoise_.push_back(raw->lumi.lumiNoise[i]);
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
  s << "LumiScalers    version: " << c.version() << std::endl;
  char line[128];

  sprintf(line," SectionNumber: %10d   StartOrbit: %10d  NumOrbits: %10d",
	  c.sectionNumber(), c.startOrbit(), c.numOrbits());
  s << line << std::endl;

  sprintf(line," Normalization: %e", c.normalization());
  s << line << std::endl;

  sprintf(line," InstantLumi:   %e   Err: %e    Qlty: %e",
	  c.instantLumi(), c.instantLumiErr(), c.instantLumiQlty());
  s << line << std::endl;

  sprintf(line," InstantETLumi: %e   Err: %e    Qlty: %e",
	  c.instantETLumi(), c.instantETLumiErr(), c.instantETLumiQlty());
  s << line << std::endl;

  int length = c.instantOccLumi().size();
  for (int i=0; i<length; i++)
  {
    sprintf(line," InstantOccLumi[%d]: %e  Err: %e  Qlty: %e",
	    i, c.instantOccLumi()[i], c.instantOccLumiErr()[i], 
	    c.instantOccLumiQlty()[i]);
    s << line << std::endl;
    sprintf(line,"      LumiNoise[%d]: %e",
	    i, c.instantOccLumi()[i],  c.lumiNoise()[i]);
    s << line << std::endl;
  }

  return s;
}
