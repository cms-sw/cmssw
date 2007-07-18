#include "CondFormats/CSCObjects/interface/CSCPedestals.h"
#include "FWCore/Utilities/interface/Exception.h"

CSCPedestals::CSCPedestals(){}
CSCPedestals::~CSCPedestals(){}

const CSCPedestals::Item & CSCPedestals::item(int cscId, int strip) const
{
  PedestalMap::const_iterator mapItr = pedestals.find(cscId);
  if(mapItr == pedestals.end())
  {
    throw cms::Exception("CSCPedestals")
      << "Cannot find CSC conditions for chamber " << cscId;
  }
  return mapItr->second.at(strip-1);
}

