#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/Framework/interface/ESHandle.h"
#include <Geometry/RPCGeometry/interface/RPCGeomServ.h>
#include <DataFormats/RPCRecHit/interface/RPCRecHit.h>
#include <DataFormats/RPCRecHit/interface/RPCRecHitCollection.h>
#include <RecoLocalMuon/RPCRecHit/interface/SIMHITtoRPC.h>
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"


SIMHITtoRPC::SIMHITtoRPC(std::vector<edm::Handle<edm::PSimHitContainer> > theSimHitContainers, const edm::EventSetup& iSetup,const edm::Event& iEvent, bool debug, int partid){
  
  edm::ESHandle<RPCGeometry> rpcGeo;
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);
  if(debug )std::cout << " The Number of sim Hits is  " << theSimHitContainers.size() <<std::endl;
  
  _ThePoints = new RPCRecHitCollection();

  std::vector<PSimHit> theSimHits;
  for (int i = 0; i < int(theSimHitContainers.size()); i++){
    theSimHits.insert(theSimHits.end(),theSimHitContainers.at(i)->begin(),theSimHitContainers.at(i)->end());
  } 
  
  if(debug )std::cout << " Looping on the SimHits"<<std::endl;
  for (std::vector<PSimHit>::const_iterator iHit = theSimHits.begin(); iHit != theSimHits.end(); iHit++){
    if(abs((*iHit).particleType())== partid){
      DetId theDetUnitId = DetId((*iHit).detUnitId());
      if(debug) std::cout << " This sim hit was produced by a particle id:"<<(*iHit).particleType()<<std::endl;
      if(debug )std::cout <<"\t Muon DetId RPC "<<DetId::Muon<<" "<<MuonSubdetId::RPC<<" "<<std::endl;
      if(debug )std::cout <<"\t SimHit   at "<<theDetUnitId.det()<<" "<<theDetUnitId.subdetId()<<" "<<std::endl;
      if(theDetUnitId.det()==DetId::Muon &&  theDetUnitId.subdetId()== MuonSubdetId::RPC){//Only RPCs
	if(debug )std::cout <<"\t RPCSimHit found "<<DetId::Muon<<" "<<MuonSubdetId::RPC<<" "<<std::endl;
	RPCDetId rollId(theDetUnitId);
	if(debug )std::cout << "\t RPCSimHit in "<<rollId<<std::endl;
	RPCRecHit RPCPoint(rollId,0,(*iHit).localPosition());
	if(debug )std::cout<<"\t \t in LocalPoint = "<<(*iHit).localPosition()<<std::endl;
	if(debug) std::cout<<"\t \t Clearing the vector"<<std::endl;	
	RPCPointVector.clear();
	if(debug) std::cout<<"\t \t \t Pushing back"<<std::endl;	
	RPCPointVector.push_back(RPCPoint); 
	if(debug) std::cout<<"\t \t \t Putting the vector"<<std::endl;	
	_ThePoints->put(rollId,RPCPointVector.begin(),RPCPointVector.end());
      }
    }
  }
}

SIMHITtoRPC::~SIMHITtoRPC(){

}
