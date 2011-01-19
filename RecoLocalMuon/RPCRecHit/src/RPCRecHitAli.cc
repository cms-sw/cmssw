// -*- C++ -*-
//
// Package:    RPCRecHitAli
// Class:      RPCRecHitAli
// 
/**\class RPCRecHitAli RPCRecHitAli.cc Analysis/RPCRecHitAli/src/RPCRecHitAli.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Camilo Andres Carrillo Montoya
//         Created:  Wed Sep 16 14:56:18 CEST 2009
// $Id: RPCRecHitAli.cc,v 1.3 2010/12/16 18:21:24 carrillo Exp $
//
//

#include "RecoLocalMuon/RPCRecHit/interface/RPCRecHitAli.h"

// system include files

// user include files

RPCRecHitAli::RPCRecHitAli(const edm::ParameterSet& iConfig)
{
  debug=iConfig.getUntrackedParameter<bool>("debug",false);
  rpcRecHitsLabel = iConfig.getParameter<edm::InputTag>("rpcRecHits");
  AlignmentinfoFile  = iConfig.getUntrackedParameter<std::string>("AliFileName","/afs/cern.ch/user/c/carrillo/segments/CMSSW_2_2_10/src/DQM/RPCMonitorModule/data/Alignment.dat"); 
  produces<RPCRecHitCollection>("RPCRecHitAligned");

  if(debug) std::cout<<"The used file for alignment is"<<AlignmentinfoFile.c_str()<<std::endl;

  std::ifstream ifin(AlignmentinfoFile.c_str());
 
 
  int rawId;
  std::string name;
  float offset;
  float rms;

  while (ifin.good()){
    ifin >>name >>rawId >> offset >> rms;
    alignmentinfo[rawId]=offset;
    if(debug) std::cout<<"rawId ="<<rawId<<" offset="<<offset<<std::endl;
  }

}


RPCRecHitAli::~RPCRecHitAli(){

}

void RPCRecHitAli::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){

  edm::Handle<RPCRecHitCollection> rpcRecHits;
  iEvent.getByLabel(rpcRecHitsLabel,rpcRecHits);

  RPCRecHitCollection::const_iterator recHit;
  
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);

  _ThePoints = new RPCRecHitCollection();

  for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
    if(dynamic_cast< RPCChamber* >( *it ) != 0 ){
      RPCChamber* ch = dynamic_cast< RPCChamber* >( *it );
      std::vector< const RPCRoll*> roles = (ch->rolls());

      for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	
	RPCDetId rollId = (*r)->id();
	
	typedef std::pair<RPCRecHitCollection::const_iterator, RPCRecHitCollection::const_iterator> rangeRecHits;
	rangeRecHits recHitCollection =  rpcRecHits->get(rollId);
	RPCRecHitCollection::const_iterator recHit;

	RPCPointVector.clear();
	for(recHit = recHitCollection.first; recHit != recHitCollection.second ; recHit++){
	  float newXPosition = 0;
	  if(alignmentinfo.find(rollId.rawId())==alignmentinfo.end()){
	    if(debug) std::cout<<"Warning the RawId = "<<rollId.rawId()<<"is not in the map"<<std::endl;
	    newXPosition = recHit->localPosition().x();
	  }else{
	    if(debug)std::cout<<"correction taking place from:"<<recHit->localPosition().x();
	    newXPosition=recHit->localPosition().x()+alignmentinfo[rollId.rawId()];
	    if(debug)std::cout<<" to:"<<newXPosition<<std::endl;
	  }
	  RPCRecHit RPCPoint(rollId,recHit->BunchX(),recHit->firstClusterStrip(),recHit->clusterSize(),LocalPoint(newXPosition,recHit->localPosition().y(),recHit->localPosition().z()),recHit->localPositionError());
	  RPCPointVector.push_back(RPCPoint);
	}
	_ThePoints->put(rollId,RPCPointVector.begin(),RPCPointVector.end());
      }
    }
  }
  
  std::auto_ptr<RPCRecHitCollection> TheAlignedRPCHits(_ThePoints);
  iEvent.put(TheAlignedRPCHits,"RPCRecHitAligned");
}

// ------------ method called once each job just before starting event loop  ------------
void 
RPCRecHitAli::beginJob(){
  
}

// ------------ method called once each job just after ending the event loop  ------------
void 
RPCRecHitAli::endJob() {
}


