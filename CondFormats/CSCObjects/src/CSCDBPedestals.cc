#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"
#include "FWCore/Utilities/interface/Exception.h"

CSCDBPedestals::CSCDBPedestals(){}
CSCDBPedestals::~CSCDBPedestals(){}

const CSCDBPedestals::Item & CSCDBPedestals::item(const CSCDetId & cscId, int strip) const
{
  CSCIndexer indexer;
  return pedestals.at( indexer.stripChannelIndex(cscId, strip) );
}

