#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"

CSCDBPedestals::CSCDBPedestals(){}
CSCDBPedestals::~CSCDBPedestals(){}

const CSCDBPedestals::Item & CSCDBPedestals::item(const CSCDetId & cscId, int strip) const
 {
  CSCIndexer indexer;
  return pedestals[ indexer.stripChannelIndex(cscId, strip)-1 ];
 }

