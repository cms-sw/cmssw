#ifndef RecoLocalMuon_RPCRecHit_DTObjectMap_h
#define RecoLocalMuon_RPCRecHit_DTObjectMap_h

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "RecoLocalMuon/RPCRecHit/src/DTStationIndex.h"

#include <set>
#include <map>

class DTObjectMap {
public:
  static DTObjectMap* GetInstance(const edm::EventSetup& iSetup);
  std::set<RPCDetId> GetRolls(DTStationIndex dtstationindex){return mapInstance->rollstoreDT[dtstationindex];}
  std::map<DTStationIndex,std::set<RPCDetId> > rollstoreDT;
  DTObjectMap(const edm::EventSetup& iSetup);

private:
  static DTObjectMap* mapInstance;
}; 

#endif // RecoLocalMuon_RPCRecHit_DTObjectMap_h
