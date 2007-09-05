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

std::ostream & operator<<(std::ostream & os, const CSCDBGains & cscDbGains)
{
  CSCIndexer indexer;
  for(int i = 0; i < cscDbGains.gains.size(); ++i)
  {
    std::pair<CSCDetId, CSCIndexer::IndexType> indexPair = indexer.detIdFromStripChannelIndex(i);
    os << indexPair.first << " strip:" << indexPair.second 
       << " slope:" << cscDbGains.gains[i].gain_slope 
       << " intercept:" << cscDbGains.gains[i].gain_intercept << "\n";
  }
  return os;
}
