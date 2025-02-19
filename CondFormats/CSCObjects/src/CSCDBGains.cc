#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"

CSCDBGains::CSCDBGains(){}
CSCDBGains::~CSCDBGains(){}

const CSCDBGains::Item & CSCDBGains::item(const CSCDetId & cscId, int strip) const
{
  CSCIndexer indexer;
  //  return gains.at( indexer.stripChannelIndex(cscId, strip)-1 ); // if we worry about range
  return gains[ indexer.stripChannelIndex(cscId, strip)-1 ]; // no worries about range!
}

std::ostream & operator<<(std::ostream & os, const CSCDBGains & cscDbGains)
{
  CSCIndexer indexer;
  for(size_t i = 0; i < cscDbGains.gains.size(); ++i)
  {
    std::pair<CSCDetId, CSCIndexer::IndexType> indexPair = indexer.detIdFromStripChannelIndex(i);
    os << indexPair.first << " strip:" << indexPair.second 
       << " slope:" << cscDbGains.gains[i].gain_slope << "\n";
  }
  return os;
}
