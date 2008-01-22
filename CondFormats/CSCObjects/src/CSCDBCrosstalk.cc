#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"

const CSCDBCrosstalk::Item & CSCDBCrosstalk::item(const CSCDetId & cscId, int strip) const
{
  CSCIndexer indexer;
  return crosstalk.at( indexer.stripChannelIndex(cscId, strip)-1 );
}

