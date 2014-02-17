// -*- C++ -*-
//
// Class:      HLTRPCFilter
// 
/**\class RPCPathChambFilter HLTRPCFilter.cc HLTrigger/special/src/HLTRPCFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Camilo Andres Carrillo Montoya
//         Created:  Thu Oct 29 11:04:22 CET 2009
// $Id: HLTRPCFilter.cc,v 1.4 2012/01/23 00:40:13 fwyzard Exp $
//
//

#include "HLTrigger/special/interface/HLTRPCFilter.h"

// system include files
#include <memory>

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

HLTRPCFilter::HLTRPCFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed

  rangestrips = iConfig.getUntrackedParameter<double>("rangestrips",1.);
  rpcRecHitsLabel = iConfig.getParameter<edm::InputTag>("rpcRecHits");
  rpcDTPointsLabel  = iConfig.getParameter<edm::InputTag>("rpcDTPoints");
  rpcCSCPointsLabel  = iConfig.getParameter<edm::InputTag>("rpcCSCPoints");
}


HLTRPCFilter::~HLTRPCFilter()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool HLTRPCFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<RPCRecHitCollection> rpcHits;
  iEvent.getByLabel(rpcRecHitsLabel,rpcHits);

  RPCRecHitCollection::const_iterator rpcPoint;
 
  if(rpcHits->begin()==rpcHits->end()){
    //std::cout<<" skiped preventing no RPC runs"<<std::endl;
    return false;
  }

  edm::Handle<RPCRecHitCollection> rpcDTPoints;
  iEvent.getByLabel(rpcDTPointsLabel,rpcDTPoints);

  edm::Handle<RPCRecHitCollection> rpcCSCPoints;
  iEvent.getByLabel(rpcCSCPointsLabel,rpcCSCPoints);

  float cluSize = 0;
  
  //DTPart

  for(rpcPoint = rpcDTPoints->begin(); rpcPoint != rpcDTPoints->end(); rpcPoint++){
    LocalPoint PointExtrapolatedRPCFrame = rpcPoint->localPosition();
    RPCDetId  rpcId = rpcPoint->rpcId();
    typedef std::pair<RPCRecHitCollection::const_iterator, RPCRecHitCollection::const_iterator> rangeRecHits;
    rangeRecHits recHitCollection =  rpcHits->get(rpcId);
    if(recHitCollection.first==recHitCollection.second){
      //std::cout<<"DT passed, no rechits for this RPCId =  "<<rpcId<<std::endl;
      return true;
    }
    float minres = 3000.;
    RPCRecHitCollection::const_iterator recHit;
    for(recHit = recHitCollection.first; recHit != recHitCollection.second ; recHit++) {
      LocalPoint recHitPos=recHit->localPosition();
      float res=PointExtrapolatedRPCFrame.x()- recHitPos.x();
      if(fabs(res)<fabs(minres)){
	minres=res;
	cluSize = recHit->clusterSize();
      }
    }
    if(fabs(minres)>=(rangestrips+cluSize*0.5)*3){ //3 is a typyical strip width for RPCs
      //std::cout<<"DT passed, RecHits but far away "<<rpcId<<std::endl;
      return true;
    }
  }

  //CSCPart

  for(rpcPoint = rpcCSCPoints->begin(); rpcPoint != rpcCSCPoints->end(); rpcPoint++){
    LocalPoint PointExtrapolatedRPCFrame = rpcPoint->localPosition();
    RPCDetId  rpcId = rpcPoint->rpcId();
    typedef std::pair<RPCRecHitCollection::const_iterator, RPCRecHitCollection::const_iterator> rangeRecHits;
    rangeRecHits recHitCollection =  rpcHits->get(rpcId);
    if(recHitCollection.first==recHitCollection.second){
      //std::cout<<"CSC passed, no rechits for this RPCId =  "<<rpcId<<std::endl;
      return true;
    }
    float minres = 3000.;
    RPCRecHitCollection::const_iterator recHit;
    for(recHit = recHitCollection.first; recHit != recHitCollection.second ; recHit++) {
      LocalPoint recHitPos=recHit->localPosition();
      float res=PointExtrapolatedRPCFrame.x()- recHitPos.x();
      if(fabs(res)<fabs(minres)){
	minres=res;
	cluSize = recHit->clusterSize();
      }
    }
    if(fabs(minres)>=(rangestrips+cluSize*0.5)*3){ //3 is a typyical strip width for RPCs
      //std::cout<<"CSC passed, RecHits but far away "<<rpcId<<std::endl;
      return true;
    }
  }

  //std::cout<<" skiped"<<std::endl;
  return false;
}


//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTRPCFilter);
