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
// $Id: RPCRecHitAli.cc,v 1.10 2010/10/19 19:34:33 wmtan Exp $
//
//

#include "RecoLocalMuon/RPCRecHit/interface/RPCRecHitAli.h"

// system include files

// user include files

RPCRecHitAli::RPCRecHitAli(const edm::ParameterSet& iConfig)
{
  debug=iConfig.getUntrackedParameter<bool>("debug",false);
  rpcRecHitsLabel = iConfig.getParameter<edm::InputTag>("rpcRecHits");
  produces<RPCRecHitCollection>("RPCRecHitAli");
}


RPCRecHitAli::~RPCRecHitAli(){

}

void RPCRecHitAli::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){

  edm::Handle<RPCRecHitCollection> rpcRecHits;
  iEvent.getByLabel(rpcRecHitsLabel,rpcRecHits);

  RPCRecHitCollection::const_iterator recHit;

  for (recHit = rpcRecHits->begin(); recHit != rpcRecHits->end(); recHit++) {
    RPCDetId rollId = (RPCDetId)(*recHit).rpcId();
    std::cout<<"RollId "<<rollId<<std::endl;
  }

  
  //std::auto_ptr<RPCRecHitCollection> TheCSCPoints(CSCClass.thePoints());  
  //iEvent.put(rpcRecHits,"RPCRecHitAli"); 
  
}

// ------------ method called once each job just before starting event loop  ------------
void 
RPCRecHitAli::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
RPCRecHitAli::endJob() {
}


