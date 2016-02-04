#include "CondFormats/CSCObjects/interface/CSCDBChipSpeedCorrection.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"

CSCDBChipSpeedCorrection::CSCDBChipSpeedCorrection(){}
CSCDBChipSpeedCorrection::~CSCDBChipSpeedCorrection(){}

const CSCDBChipSpeedCorrection::Item & CSCDBChipSpeedCorrection::item(const CSCDetId & cscId, int chip) const
 {
  CSCIndexer indexer;
  return chipSpeedCorr[ indexer.chipIndex(cscId, chip)-1 ]; // no worries about range!
 }

