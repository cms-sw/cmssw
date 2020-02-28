#include "CondFormats/CSCObjects/interface/CSCPedestals.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"
#include "FWCore/Utilities/interface/Exception.h"

CSCPedestals::CSCPedestals() {}
CSCPedestals::~CSCPedestals() {}

const CSCPedestals::Item& CSCPedestals::item(const CSCDetId& cscId, int strip) const {
  CSCIndexer indexer;
  PedestalMap::const_iterator mapItr = pedestals.find(indexer.dbIndex(cscId, strip));
  if (mapItr == pedestals.end()) {
    throw cms::Exception("CSCPedestals") << "Cannot find CSC conditions for chamber " << cscId;
  }
  return mapItr->second.at(strip - 1);
}
