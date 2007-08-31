#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"
#include "FWCore/Utilities/interface/Exception.h"

CSCDBGains::CSCDBGains(){}
CSCDBGains::~CSCDBGains(){}

const CSCDBGains::Item & CSCDBGains::item(const CSCDetId & cscId, int strip) const
{
  CSCIndexer indexer;
  return gains.at( indexer.stripChannelIndex(cscId, strip) );
}

