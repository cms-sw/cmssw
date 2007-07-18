#include "CondFormats/CSCObjects/interface/CSCGains.h"
#include "FWCore/Utilities/interface/Exception.h"

CSCGains::CSCGains(){}
CSCGains::~CSCGains(){}

const CSCGains::Item & CSCGains::item(int cscId, int strip) const
{
  GainsMap::const_iterator mapItr = gains.find(cscId);
  if(mapItr == gains.end())
  {
    throw cms::Exception("CSCGains")
      << "Cannot find CSC conditions for chamber " << cscId;
  }
  return mapItr->second.at(strip-1);
}

