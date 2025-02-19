#include "CondFormats/CSCObjects/interface/CSCDBGasGainCorrection.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"

CSCDBGasGainCorrection::CSCDBGasGainCorrection(){}
CSCDBGasGainCorrection::~CSCDBGasGainCorrection(){}

const CSCDBGasGainCorrection::Item & CSCDBGasGainCorrection::item(const CSCDetId & cscId, int strip, int wire) const
 {
  CSCIndexer indexer;
  //note the transformation here from database index (starting from 1) to c++ indexing (starting from 0)
  return gasGainCorr[ indexer.gasGainIndex(cscId, strip, wire)-1 ]; // no worries about range!
 }

