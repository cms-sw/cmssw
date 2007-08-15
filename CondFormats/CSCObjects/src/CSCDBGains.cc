#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include "FWCore/Utilities/interface/Exception.h"

CSCDBGains::CSCDBGains(){}
CSCDBGains::~CSCDBGains(){}
/*
const CSCDBGains::Item & CSCGains::item(int cscId, int strip) const
{
  GainsMap::const_iterator mapItr = gains.find(cscId);
  if(mapItr == gains.end())
  {
    throw cms::Exception("CSCGains")
      << "Cannot find CSC conditions for chamber " << cscId;
  }
  return mapItr->second.at(strip-1);

}
*/
