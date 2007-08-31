#include "CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"
#include "FWCore/Utilities/interface/Exception.h"

CSCDBNoiseMatrix::CSCDBNoiseMatrix(){}
CSCDBNoiseMatrix::~CSCDBNoiseMatrix(){}

const CSCDBNoiseMatrix::Item & CSCDBNoiseMatrix::item(const CSCDetId & cscId, int strip) const
{
  CSCIndexer indexer;
  return matrix.at( indexer.stripChannelIndex(cscId, strip) );
}


