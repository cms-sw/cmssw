#ifndef RecoLocalMuon_RPCRecHit_CSCObjectMap_h
#define RecoLocalMuon_RPCRecHit_CSCObjectMap_h

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "RecoLocalMuon/RPCRecHit/src/CSCStationIndex.h"

#include <set>
#include <map>

class CSCObjectMap{
public:
  static CSCObjectMap* GetInstance(const edm::EventSetup& iSetup);
  std::set<RPCDetId> GetRolls(CSCStationIndex cscstationindex){return mapInstance->rollstoreCSC[cscstationindex];}
  std::map<CSCStationIndex,std::set<RPCDetId> > rollstoreCSC;
  CSCObjectMap(const edm::EventSetup& iSetup);

private:
  static CSCObjectMap* mapInstance;
}; 

#endif // RecoLocalMuon_RPCRecHit_CSCObjectMap_h
