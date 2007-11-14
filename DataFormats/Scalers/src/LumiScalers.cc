
/*
 *   File: DataFormats/Scalers/src/LumiScalers.cc   (W.Badgett)
 */

#include "DataFormats/Scalers/interface/LumiScalers.h"

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
{ }

LumiScalers::~LumiScalers() { } 


/// Pretty-print operator for LumiScalers
std::ostream& operator<<(std::ostream& s, const LumiScalers& c) 
{
  s << " LumiScalers: ";
  return s;
}
