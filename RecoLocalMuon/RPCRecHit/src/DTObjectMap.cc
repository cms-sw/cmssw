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

DTObjectMap::DTObjectMap(MuonGeometryRecord const& record)
{
  edm::ESHandle<RPCGeometry> rpcGeo;
  record.get(rpcGeo);

  edm::ESHandle<DTGeometry> dtGeo;
  record.get(dtGeo);
  
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
	  if (rollstore.find(ind)!=rollstore.end()) myrolls=rollstore[ind];
	  myrolls.insert(rpcId);
	  rollstore[ind]=myrolls;
	}
      }
    }
  }
}

std::set<RPCDetId> const& DTObjectMap::getRolls(DTStationIndex index) const
{
  // FIXME
  // the present inplementation allows for NOT finding the given index in the map;
  // a muon expert should check that this is the intended behaviour.
  static const std::set<RPCDetId> empty;
  return (rollstore.find(index) == rollstore.end()) ? empty : rollstore.at(index);
}

// register the class with the typelookup system used by the EventSetup
#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(DTObjectMap);
