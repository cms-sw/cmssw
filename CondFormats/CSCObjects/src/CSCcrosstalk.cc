#include "CondFormats/CSCObjects/interface/CSCcrosstalk.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"
#include "FWCore/Utilities/interface/Exception.h"

const CSCcrosstalk::Item& CSCcrosstalk::item(const CSCDetId& cscId, int strip) const {
  CSCIndexer indexer;
  CrosstalkMap::const_iterator mapItr = crosstalk.find(indexer.dbIndex(cscId, strip));
  if (mapItr == crosstalk.end()) {
    throw cms::Exception("CSCCrosstalk") << "Cannot find CSC conditions for chamber " << cscId;
  }
  return mapItr->second.at(strip - 1);
}
