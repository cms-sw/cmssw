#ifndef RecoLocalMuon_RPCRecHit_CSCObjectMap_h
#define RecoLocalMuon_RPCRecHit_CSCObjectMap_h

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "RecoLocalMuon/RPCRecHit/src/CSCStationIndex.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"

#include <set>
#include <map>

class CSCObjectMap {
public:
  CSCObjectMap(RPCGeometry const& rpcGeom);

  std::set<RPCDetId> const& getRolls(CSCStationIndex index) const;

private:
  std::map<CSCStationIndex, std::set<RPCDetId>> rollstore;
};

#endif  // RecoLocalMuon_RPCRecHit_CSCObjectMap_h
