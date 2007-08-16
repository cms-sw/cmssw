#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"
#include "FWCore/Utilities/interface/Exception.h"
/*
const CSCDBCrosstalk::Item & CSCDBCrosstalk::item(int cscId, int strip) const
{
  CrosstalkContainer::const_iterator Itr = crosstalk.find(cscId);
  if(Itr == crosstalk.end())
  {
    throw cms::Exception("CSCDBCrosstalk") 
      << "Cannot find CSC conditions for chamber " << cscId;
  }
  return Itr->second.at(strip-1);
}
*/
