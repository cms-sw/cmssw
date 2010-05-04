#include "CondFormats/CSCObjects/interface/CSCDBChipSpeedCorrection.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"

CSCDBChipSpeedCorrection::CSCDBChipSpeedCorrection(){}
CSCDBChipSpeedCorrection::~CSCDBChipSpeedCorrection(){}

const CSCDBChipSpeedCorrection::Item & CSCDBChipSpeedCorrection::item(const CSCDetId & cscId, int strip) const
 {
  CSCIndexer indexer;
  //  return pedestals.at( indexer.stripChannelIndex(cscId, strip)-1 ); // if we worry about range
  //chamge to CHIP below!
  return chipSpeedCorr[ indexer.stripChannelIndex(cscId, strip)-1 ]; // no worries about range!
 }

