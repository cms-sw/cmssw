#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "Geometry/RPCGeometry/interface/RPCChamber.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "RecoLocalMuon/RPCRecHit/src/DTObjectMap.h"
#include "RecoLocalMuon/RPCRecHit/src/DTStationIndex.h"

DTObjectMap* DTObjectMap::mapInstance = NULL;

DTObjectMap* DTObjectMap::GetInstance(const edm::EventSetup& iSetup){
  if (mapInstance == NULL){
    mapInstance = new DTObjectMap(iSetup);
  }
  return mapInstance;
}

DTObjectMap::DTObjectMap(const edm::EventSetup& iSetup){
  edm::ESHandle<RPCGeometry> rpcGeo;
  edm::ESHandle<DTGeometry> dtGeo;
  
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);
  iSetup.get<MuonGeometryRecord>().get(dtGeo);
  
  for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
    if(dynamic_cast<const RPCChamber* >( *it ) != 0 ){
      auto ch = dynamic_cast<const RPCChamber* >( *it ); 
      std::vector< const RPCRoll*> roles = (ch->rolls());
      for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	RPCDetId rpcId = (*r)->id();
	int region=rpcId.region();
	if(region==0){
	  int wheel=rpcId.ring();
	  int sector=rpcId.sector();
	  int station=rpcId.station();
	  DTStationIndex ind(region,wheel,sector,station);
	  std::set<RPCDetId> myrolls;
	  if (rollstoreDT.find(ind)!=rollstoreDT.end()) myrolls=rollstoreDT[ind];
	  myrolls.insert(rpcId);
	  rollstoreDT[ind]=myrolls;
	}
      }
    }
  }
}
