#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"

CSCDBPedestals::CSCDBPedestals(){}
CSCDBPedestals::~CSCDBPedestals(){}

const CSCDBPedestals::Item & CSCDBPedestals::item(const CSCDetId & cscId, int strip) const
 {
  CSCIndexer indexer;
  //  return pedestals.at( indexer.stripChannelIndex(cscId, strip)-1 ); // if we worry about range
  return pedestals[ indexer.stripChannelIndex(cscId, strip)-1 ]; // no worries about range!
 }

