#include "CondFormats/CSCObjects/interface/CSCGains.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"
#include "FWCore/Utilities/interface/Exception.h"

CSCGains::CSCGains(){}
CSCGains::~CSCGains(){}

const CSCGains::Item & CSCGains::item(const CSCDetId & cscId, int strip) const
{
  CSCIndexer indexer;
  GainsMap::const_iterator mapItr = gains.find( indexer.dbIndex(cscId, strip) );
  if(mapItr == gains.end())
  {
    throw cms::Exception("CSCGains")
      << "Cannot find CSC conditions for chamber " << cscId;
  }
  return mapItr->second.at(strip-1);
}

