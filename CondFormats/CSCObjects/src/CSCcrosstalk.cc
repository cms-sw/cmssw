#include "CondFormats/CSCObjects/interface/CSCcrosstalk.h"
#include "FWCore/Utilities/interface/Exception.h"

const CSCcrosstalk::Item & CSCcrosstalk::item(int cscId, int strip) const
{
  CrosstalkMap::const_iterator mapItr = crosstalk.find(cscId);
  if(mapItr == crosstalk.end())
  {
    throw cms::Exception("CSCCrosstalk") 
      << "Cannot find CSC conditions for chamber " << cscId;
  }
  return mapItr->second.at(strip-1);
}

