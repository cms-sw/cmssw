#ifndef RecoLocalMuon_RPCRecHit_DTObjectMap_h
#define RecoLocalMuon_RPCRecHit_DTObjectMap_h

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "RecoLocalMuon/RPCRecHit/src/DTStationIndex.h"

#include <set>
#include <map>

class DTObjectMap {
public:
  DTObjectMap(MuonGeometryRecord const& record);

  std::set<RPCDetId> const & getRolls(DTStationIndex index) const;

private:
  std::map<DTStationIndex, std::set<RPCDetId>> rollstore;
}; 

#endif // RecoLocalMuon_RPCRecHit_DTObjectMap_h
