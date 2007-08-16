#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "FWCore/Utilities/interface/Exception.h"

CSCDBPedestals::CSCDBPedestals(){}
CSCDBPedestals::~CSCDBPedestals(){}
/*
const CSCDBPedestals::Item & CSCDBPedestals::item(int cscId, int strip) const
{
  PedestalContainer::const_iterator Itr = pedestals.find(cscId);
  if(Itr == pedestals.end())
  {
    throw cms::Exception("CSCDBPedestals")
      << "Cannot find CSC conditions for chamber " << cscId;
  }
  return Itr->second.at(strip-1);
}

*/
