#include "CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"

#include <iostream>
CSCDBNoiseMatrix::CSCDBNoiseMatrix(){}
CSCDBNoiseMatrix::~CSCDBNoiseMatrix(){}

const CSCDBNoiseMatrix::Item & CSCDBNoiseMatrix::item(const CSCDetId & cscId, int strip) const
{
  CSCIndexer indexer;
  //  return matrix.at( indexer.stripChannelIndex(cscId, strip)-1 ); // if we worry about range
  return matrix[ indexer.stripChannelIndex(cscId, strip)-1 ]; // no worries about range!
}


