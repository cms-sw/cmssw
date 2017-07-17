#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "Geometry/RPCGeometry/interface/RPCChamber.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "RecoLocalMuon/RPCRecHit/src/CSCObjectMap.h"
#include "RecoLocalMuon/RPCRecHit/src/CSCStationIndex.h"

CSCObjectMap::CSCObjectMap(MuonGeometryRecord const& record){
  edm::ESHandle<RPCGeometry> rpcGeo;
  record.get(rpcGeo);

  edm::ESHandle<CSCGeometry> cscGeo;
  record.get(cscGeo);
  
  for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
    if(dynamic_cast< const RPCChamber* >( *it ) != 0 ){
      auto ch = dynamic_cast< const RPCChamber* >( *it ); 
      std::vector< const RPCRoll*> roles = (ch->rolls());
      for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	RPCDetId rpcId = (*r)->id();
	int region=rpcId.region();
	if(region!=0){
	  int station=rpcId.station();
          int ring=rpcId.ring();
          int cscring=ring;
          int cscstation=station;
	  RPCGeomServ rpcsrv(rpcId);
	  int rpcsegment = rpcsrv.segment();
	  int cscchamber = rpcsegment; //FIX THIS ACCORDING TO RPCGeomServ::segment()Definition
          if((station==2||station==3)&&ring==3){//Adding Ring 3 of RPC to the CSC Ring 2
            cscring = 2;
          }
	  CSCStationIndex ind(region,cscstation,cscring,cscchamber);
          std::set<RPCDetId> myrolls;
	  if (rollstore.find(ind)!=rollstore.end()) myrolls=rollstore[ind];
	  myrolls.insert(rpcId);
          rollstore[ind]=myrolls;
	}
      }
    }
  }
}

std::set<RPCDetId> const& CSCObjectMap::getRolls(CSCStationIndex index) const
{
  // FIXME
  // the present inplementation allows for NOT finding the given index in the map;
  // a muon expert should check that this is the intended behaviour.
  static const std::set<RPCDetId> empty;
  return (rollstore.find(index) == rollstore.end()) ? empty : rollstore.at(index);
}

// register the class with the typelookup system used by the EventSetup
#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(CSCObjectMap);
