#ifndef RecoLocalMuon_RPCRecHit_CSCObjectMap_h
#define RecoLocalMuon_RPCRecHit_CSCObjectMap_h

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "RecoLocalMuon/RPCRecHit/src/CSCStationIndex.h"

#include <set>
#include <map>

class CSCObjectMap {
public:
  CSCObjectMap(MuonGeometryRecord const& record);

  std::set<RPCDetId> const& getRolls(CSCStationIndex index) const;

private:
  std::map<CSCStationIndex,std::set<RPCDetId>> rollstore;
}; 

#endif // RecoLocalMuon_RPCRecHit_CSCObjectMap_h
