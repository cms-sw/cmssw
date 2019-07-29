#ifndef RecoLocalMuon_RPCRecHit_DTObjectMap_h
#define RecoLocalMuon_RPCRecHit_DTObjectMap_h

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "RecoLocalMuon/RPCRecHit/src/DTStationIndex.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"

#include <set>
#include <map>

class DTObjectMap {
public:
  DTObjectMap(RPCGeometry const& rpcGeometry);

  std::set<RPCDetId> const& getRolls(DTStationIndex index) const;

private:
  std::map<DTStationIndex, std::set<RPCDetId>> rollstore;
};

#endif  // RecoLocalMuon_RPCRecHit_DTObjectMap_h
