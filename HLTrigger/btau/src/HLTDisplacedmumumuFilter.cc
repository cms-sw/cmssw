#include <iostream>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

#include "HLTDisplacedmumumuFilter.h"
#include "TMath.h"

//
// constructors and destructor
//
HLTDisplacedmumumuFilter::HLTDisplacedmumumuFilter(const edm::ParameterSet& iConfig):
 
  fastAccept_ (iConfig.getParameter<bool>("FastAccept")),
  minLxySignificance_ (iConfig.getParameter<double>("MinLxySignificance")),
  maxLxySignificance_ (iConfig.existsAs<double>("MaxLxySignificance") ? iConfig.getParameter<double>("MaxLxySignificance") : -1. ),
  maxNormalisedChi2_ (iConfig.getParameter<double>("MaxNormalisedChi2")), 
  minVtxProbability_ (iConfig.getParameter<double>("MinVtxProbability")),
  minCosinePointingAngle_ (iConfig.getParameter<double>("MinCosinePointingAngle")),
  saveTags_ (iConfig.getParameter<bool>("saveTags")),
  DisplacedVertexTag_(iConfig.getParameter<edm::InputTag>("DisplacedVertexTag")),
  beamSpotTag_ (iConfig.getParameter<edm::InputTag> ("BeamSpotTag")),
  MuonTag_ (iConfig.getParameter<edm::InputTag>("MuonTag"))
 
{

  produces<trigger::TriggerFilterObjectWithRefs>();
}


HLTDisplacedmumumuFilter::~HLTDisplacedmumumuFilter()
{

}


// ------------ method called once each job just before starting event loop  ------------
void HLTDisplacedmumumuFilter::beginJob()
{

}

// ------------ method called once each job just after ending the event loop  ------------
void HLTDisplacedmumumuFilter::endJob() 
{
 	
}

// ------------ method called on each new Event  ------------
bool HLTDisplacedmumumuFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{


  // get beam spot
  reco::BeamSpot vertexBeamSpot;
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByLabel(beamSpotTag_,recoBeamSpotHandle);
  vertexBeamSpot = *recoBeamSpotHandle;
 
  
  // get displaced vertices
  reco::VertexCollection displacedVertexColl;
  edm::Handle<reco::VertexCollection> displacedVertexCollHandle;
  bool foundVertexColl = iEvent.getByLabel(DisplacedVertexTag_, displacedVertexCollHandle);
  if(foundVertexColl) displacedVertexColl = *displacedVertexCollHandle;
  
 
  // get muon collection
  edm::Handle<reco::RecoChargedCandidateCollection> mucands;
  iEvent.getByLabel (MuonTag_,mucands);
 
  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.
  

  // Ref to Candidate object to be recorded in filter object
  reco::RecoChargedCandidateRef ref1;
  reco::RecoChargedCandidateRef ref2;
  reco::RecoChargedCandidateRef ref3;

  std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterobject (new trigger::TriggerFilterObjectWithRefs(path(),module()));
  if(saveTags_) 	  filterobject->addCollectionTag(MuonTag_);

  
  bool triggered = false;
 
  // loop over vertex collection
  for(reco::VertexCollection::iterator it = displacedVertexColl.begin(); it!= displacedVertexColl.end(); it++){
          reco::Vertex displacedVertex = *it;
        
          // check if the vertex actually consists of exactly two muon tracks, throw exception if not
          if(displacedVertex.tracksSize() != 3)  throw cms::Exception("BadLogic") << "HLTDisplacedmumumuFilter: ERROR: the Jpsi vertex must have exactly three muons by definition. It now has n muons = "
        									    << displacedVertex.tracksSize() << std::endl;
       
          float normChi2 = displacedVertex.normalizedChi2();
	  if (normChi2 > maxNormalisedChi2_) continue;

	  double vtxProb = 0.0;
	  if( (displacedVertex.chi2()>=0.0) && (displacedVertex.ndof()>0) ) vtxProb = TMath::Prob(displacedVertex.chi2(), displacedVertex.ndof() );
	  if (vtxProb < minVtxProbability_) continue;

          // get the two muons from the vertex
          reco::Vertex::trackRef_iterator trackIt =  displacedVertex.tracks_begin();
          reco::TrackRef vertextkRef1 =  (*trackIt).castTo<reco::TrackRef>() ;
          // the second one
          trackIt++;
          reco::TrackRef vertextkRef2 =  (*trackIt).castTo<reco::TrackRef>();
          // the second one
          trackIt++;
          reco::TrackRef vertextkRef3 =  (*trackIt).castTo<reco::TrackRef>();
          
	  reco::RecoChargedCandidateCollection::const_iterator cand1;
	  reco::RecoChargedCandidateCollection::const_iterator cand2;	  
	  reco::RecoChargedCandidateCollection::const_iterator cand3;	  

	  // first find these two tracks in the muon collection
	  int iFoundRefs = 0;
	  for (reco::RecoChargedCandidateCollection::const_iterator cand=mucands->begin(); cand!=mucands->end(); cand++) {
	    reco::TrackRef tkRef = cand->get<reco::TrackRef>();
	    if(tkRef == vertextkRef1) {cand1 = cand; iFoundRefs++;}
	    if(tkRef == vertextkRef2) {cand2 = cand; iFoundRefs++;}
	    if(tkRef == vertextkRef3) {cand3 = cand; iFoundRefs++;}
	  }
	  if(iFoundRefs != 3) throw cms::Exception("BadLogic") << "HLTDisplacedmumumuFilter: ERROR: the Jpsi vertex must have exactly two muons by definition."  << std::endl; 

          // calculate two-track transverse momentum
          math::XYZVector pperp(cand1->px() + cand2->px() + cand3->px(),
				cand1->py() + cand2->py() + cand3->py(),
        			  0.);
          
        
	  reco::Vertex::Point vpoint=displacedVertex.position();
	  //translate to global point, should be improved
	  GlobalPoint secondaryVertex (vpoint.x(), vpoint.y(), vpoint.z());

	  reco::Vertex::Error verr = displacedVertex.error();
	  // translate to global error, should be improved
	  GlobalError err(verr.At(0,0), verr.At(1,0), verr.At(1,1), verr.At(2,0), verr.At(2,1), verr.At(2,2) );

	  GlobalPoint displacementFromBeamspot( -1*((vertexBeamSpot.x0() -  secondaryVertex.x()) +  (secondaryVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dxdz()),
        					  -1*((vertexBeamSpot.y0() - secondaryVertex.y())+  (secondaryVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dydz()), 0);
        
          float lxy = displacementFromBeamspot.perp();
          float lxyerr = sqrt(err.rerr(displacementFromBeamspot));
        
        
          //calculate the angle between the decay length and the mumumu momentum
	  reco::Vertex::Point vperp(displacementFromBeamspot.x(),displacementFromBeamspot.y(),0.);
         
          float cosAlpha = vperp.Dot(pperp)/(vperp.R()*pperp.R());
        
          // check thresholds
          if (cosAlpha < minCosinePointingAngle_) continue;
          if (minLxySignificance_ > 0. && lxy/lxyerr < minLxySignificance_) continue;
	  if (maxLxySignificance_ > 0. && lxy/lxyerr > maxLxySignificance_) continue; 
	  triggered = true;
 
	  // now add the muons that passed to the filter object
	  
	  ref1=reco::RecoChargedCandidateRef( edm::Ref<reco::RecoChargedCandidateCollection> 	(mucands,distance(mucands->begin(), cand1)));
	  filterobject->addObject(trigger::TriggerMuon,ref1);
	  
	  ref2=reco::RecoChargedCandidateRef( edm::Ref<reco::RecoChargedCandidateCollection> (mucands,distance(mucands->begin(),cand2 )));
	  filterobject->addObject(trigger::TriggerMuon,ref2);

	  ref3=reco::RecoChargedCandidateRef( edm::Ref<reco::RecoChargedCandidateCollection> (mucands,distance(mucands->begin(),cand3 )));
	  filterobject->addObject(trigger::TriggerMuon,ref3);
  }


  // put filter object into the Event
  
  iEvent.put(filterobject);
  LogDebug("HLTDisplacedMumumuFilter") << " >>>>> Result of HLTDisplacedMumumuFilter is "<< triggered <<std::endl; 

  return triggered;
}

