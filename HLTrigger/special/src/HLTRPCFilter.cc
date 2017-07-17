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

HLTRPCFilter::HLTRPCFilter(const edm::ParameterSet& config) :
  rpcRecHitsToken(   consumes<RPCRecHitCollection>( config.getParameter<edm::InputTag>("rpcRecHits") ) ),
  rpcDTPointsToken(  consumes<RPCRecHitCollection>( config.getParameter<edm::InputTag>("rpcDTPoints") ) ),
  rpcCSCPointsToken( consumes<RPCRecHitCollection>( config.getParameter<edm::InputTag>("rpcCSCPoints") ) ),
  rangestrips( config.getUntrackedParameter<double>("rangestrips", 1.) )
{
}


HLTRPCFilter::~HLTRPCFilter()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


void
HLTRPCFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("rpcRecHits",edm::InputTag("hltRpcRecHits"));
  desc.add<edm::InputTag>("rpcDTPoints",edm::InputTag("rpcPointProducer","RPCDTExtrapolatedPoints"));
  desc.add<edm::InputTag>("rpcCSCPoints",edm::InputTag("rpcPointProducer","RPCCSCExtrapolatedPoints"));
  desc.addUntracked<double>("rangestrips",4.0);
  descriptions.add("hltRPCFilter",desc);
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool HLTRPCFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
  edm::Handle<RPCRecHitCollection> rpcHits;
  iEvent.getByToken(rpcRecHitsToken,rpcHits);

  RPCRecHitCollection::const_iterator rpcPoint;
 
  if(rpcHits->begin()==rpcHits->end()){
    //std::cout<<" skipped preventing no RPC runs"<<std::endl;
    return false;
  }

  edm::Handle<RPCRecHitCollection> rpcDTPoints;
  iEvent.getByToken(rpcDTPointsToken,rpcDTPoints);

  edm::Handle<RPCRecHitCollection> rpcCSCPoints;
  iEvent.getByToken(rpcCSCPointsToken,rpcCSCPoints);

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
