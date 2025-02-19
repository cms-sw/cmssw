// -*- C++ -*-
//
// Class:      HLTRPCTrigNoSyncFilter
// 
/**\class RPCPathChambFilter HLTRPCTrigNoSyncFilter.cc HLTrigger/special/src/HLTRPCTrigNoSyncFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Camilo Andres Carrillo Montoya
//         Created:  Thu Oct 29 11:04:22 CET 2009
// $Id: HLTRPCTrigNoSyncFilter.cc,v 1.4 2012/01/21 15:00:22 fwyzard Exp $
//
//

#include "HLTrigger/special/interface/HLTRPCTrigNoSyncFilter.h"

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

typedef struct {
  int id;
  int bx;
  GlobalPoint gp;
} RPC4DHit;

bool bigmag(const RPC4DHit &Point1,const RPC4DHit &Point2){
  if((Point2).gp.mag() > (Point1).gp.mag()) return true;
  else return false;
}

HLTRPCTrigNoSyncFilter::HLTRPCTrigNoSyncFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) 
{
  //now do what ever initialization is needed
  m_GMTInputTag =iConfig.getParameter< edm::InputTag >("GMTInputTag");
  rpcRecHitsLabel = iConfig.getParameter<edm::InputTag>("rpcRecHits");  
}


HLTRPCTrigNoSyncFilter::~HLTRPCTrigNoSyncFilter()
{
    // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool HLTRPCTrigNoSyncFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct){

  std::vector<RPC4DHit> GlobalRPC4DHits;
  std::vector<RPC4DHit> GlobalRPC4DHitsNoBx0;

  edm::Handle<RPCRecHitCollection> rpcRecHits;
  
  //std::cout <<"\t Getting the RPC RecHits"<<std::endl;

  iEvent.getByLabel(rpcRecHitsLabel,rpcRecHits);
  
  if(!rpcRecHits.isValid()){
    //std::cout<<"no valid RPC Collection"<<std::endl;
    //std::cout<<"event skipped"<<std::endl;
    return false;
  }
  if(rpcRecHits->begin() == rpcRecHits->end()){
    //std::cout<<"no RPC Hits in this event"<<std::endl;
    //std::cout<<"event skipped"<<std::endl;
    return false;
  }
  
  RPCRecHitCollection::const_iterator recHit;

  edm::ESHandle<RPCGeometry> rpcGeo;
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);

  int k = 0;

  for (recHit = rpcRecHits->begin(); recHit != rpcRecHits->end(); recHit++){
    RPCDetId rollId = (RPCDetId)(*recHit).rpcId();
    LocalPoint recHitPos=recHit->localPosition();    
    const RPCRoll* rollasociated = rpcGeo->roll(rollId);
    const BoundPlane & RPCSurface = rollasociated->surface(); 
    GlobalPoint RecHitInGlobal = RPCSurface.toGlobal(recHitPos);
    
    int BX = recHit->BunchX();
    //std::cout<<"\t \t We have an RPC Rec Hit! bx="<<BX<<" Roll="<<rollId<<" Global Position="<<RecHitInGlobal<<std::endl;
    
    RPC4DHit ThisHit;
    ThisHit.bx =  BX;
    ThisHit.gp = RecHitInGlobal;
    ThisHit.id = k;
    GlobalRPC4DHits.push_back(ThisHit);
    if(BX!=0)GlobalRPC4DHitsNoBx0.push_back(ThisHit);
    k++;
  }

  if(GlobalRPC4DHitsNoBx0.size()==0){
    //std::cout<<"all RPC Hits are syncrhonized"<<std::endl;
    //std::cout<<"event skipped"<<std::endl;
    return false;
  }

  if(GlobalRPC4DHitsNoBx0.size()>100){
    //std::cout<<"too many rpcHits preventing HLT eternal loop"<<std::endl;
    //std::cout<<"event skipped"<<std::endl;
    return false;
  }
   
  edm::Handle<L1MuGMTReadoutCollection> gmtrc_handle;
  iEvent.getByLabel(m_GMTInputTag,gmtrc_handle);
  
  std::vector<L1MuGMTExtendedCand> gmt_candidates = (*gmtrc_handle).getRecord().getGMTCands();
  
  std::vector<L1MuGMTExtendedCand>::const_iterator candidate;
  //std::cout<<"The number of GMT Candidates in this event is "<<gmt_candidates.size()<<std::endl;

  if(gmt_candidates.size()==0){
    //std::cout<<"event skipped no gmt candidates"<<std::endl;
    return false;
  }
  
  for(candidate=gmt_candidates.begin(); candidate!=gmt_candidates.end(); candidate++) {
    int qual = candidate->quality();
    //std::cout<<"quality of this GMT Candidate (should be >5)= "<<qual<<std::endl;
    if(qual < 5) continue;
    float eta=candidate->etaValue();
    float phi=candidate->phiValue();      
    
    //std::cout<<" phi="<<phi<<" eta="<<eta<<std::endl;
    
    //Creating container in this etaphi direction;
    
    std::vector<RPC4DHit> PointsForGMT;
    
    for(std::vector<RPC4DHit>::iterator Point = GlobalRPC4DHitsNoBx0.begin(); Point!=GlobalRPC4DHitsNoBx0.end(); ++Point){ 
      float phiP = Point->gp.phi();
      float etaP = Point->gp.eta();
      //std::cout<<"\t \t GMT   phi="<<phi<<" eta="<<eta<<std::endl;
      //std::cout<<"\t \t Point phi="<<phiP<<" eta="<< etaP<<std::endl;
      //std::cout<<"\t \t "<<fabs(phi-phiP)<<" < 0,1? "<<fabs(eta-etaP)<<" < 0.20 ?"<<std::endl;
      if(fabs(phi-phiP) <=0.1 && fabs(eta-etaP)<=0.20){
	PointsForGMT.push_back(*Point);
	//std::cout<<"\t \t match!"<<std::endl;
      }
    }
    
    //std::cout<<"\t Points matching this GMT="<<PointsForGMT.size()<<std::endl;

    if(PointsForGMT.size()<1){
      //std::cout<<"\t Not enough RPCRecHits not syncrhonized for this GMT!!!"<<std::endl;
      continue;
    }
      
    std::sort(PointsForGMT.begin(), PointsForGMT.end(), bigmag);

    //std::cout<<"GMT candidate phi="<<phi<<std::endl;
    //std::cout<<"GMT candidate eta="<<eta<<std::endl;

    int lastbx=-7;
    bool outOfTime = false;
    bool incr = true;
    bool anydifferentzero = true;
    bool anydifferentone = true;
    
    //std::cout<<"\t \t loop on the RPCHit4D!!!"<<std::endl;
    for(std::vector<RPC4DHit>::iterator point = PointsForGMT.begin(); point < PointsForGMT.end(); ++point) {
      //float r=point->gp.mag();
      outOfTime |= (point->bx!=0); //condition 1: at least one measurement must have BX!=0
      incr &= (point->bx>=lastbx); //condition 2: BX must be increase when going inside-out.
      anydifferentzero &= (!point->bx==0); //to check one knee withoutzeros
      anydifferentone &= (!point->bx==1); //to check one knee withoutones
      lastbx = point->bx;
      //std::cout<<"\t \t  r="<<r<<" phi="<<point->gp.phi()<<" eta="<<point->gp.eta()<<" bx="<<point->bx<<" outOfTime"<<outOfTime<<" incr"<<incr<<" anydifferentzero"<<anydifferentzero<<std::endl;
    }
    //std::cout<<"\t \t";
    
    //for(std::vector<RPC4DHit>::iterator point = PointsForGMT.begin(); point < PointsForGMT.end(); ++point) {
      //std::cout<<point->bx;
    //}
    //std::cout<<std::endl;
    
    bool Candidate = (outOfTime&&incr);
    
    if(Candidate){ 
      //std::cout<<" Event Passed We found an strange trigger Candidate phi="<<phi<<" eta="<<eta<<std::endl;
      return true;
    }
  }
  
  //std::cout<<"event skipped rechits out of time but not related with a GMT "<<std::endl;
  return false;
}

// ------------ method called once each job just before starting event loop  ------------
void 
HLTRPCTrigNoSyncFilter::beginJob(){
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HLTRPCTrigNoSyncFilter::endJob(){
}

//define this as a plug-in
DEFINE_FWK_MODULE(HLTRPCTrigNoSyncFilter);
