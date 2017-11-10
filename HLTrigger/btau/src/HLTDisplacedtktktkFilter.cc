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

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "HLTrigger/btau/src/HLTDisplacedtktktkFilter.h"
#include "TMath.h"

//
// constructors and destructor
//
HLTDisplacedtktktkFilter::HLTDisplacedtktktkFilter(const edm::ParameterSet& iConfig) :
  HLTFilter(iConfig),
  fastAccept_ (iConfig.getParameter<bool>("FastAccept")),
  minLxySignificance_ (iConfig.getParameter<double>("MinLxySignificance")),
  maxLxySignificance_ (iConfig.getParameter<double>("MaxLxySignificance")),
  maxNormalisedChi2_ (iConfig.getParameter<double>("MaxNormalisedChi2")),
  minVtxProbability_ (iConfig.getParameter<double>("MinVtxProbability")),
  minCosinePointingAngle_ (iConfig.getParameter<double>("MinCosinePointingAngle")),
  triggerTypeDaughters_(iConfig.getParameter<int>("triggerTypeDaughters")),
  DisplacedVertexTag_(iConfig.getParameter<edm::InputTag>("DisplacedVertexTag")),
  DisplacedVertexToken_(consumes<reco::VertexCollection>(DisplacedVertexTag_)),
  beamSpotTag_ (iConfig.getParameter<edm::InputTag> ("BeamSpotTag")),
  beamSpotToken_(consumes<reco::BeamSpot>(beamSpotTag_)),
  TrackTag_ (iConfig.getParameter<edm::InputTag>("TrackTag")),
  TrackToken_(consumes<reco::RecoChargedCandidateCollection>(TrackTag_))
{
}


HLTDisplacedtktktkFilter::~HLTDisplacedtktktkFilter() = default;

void HLTDisplacedtktktkFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<bool>("FastAccept",false);
  desc.add<double>("MinLxySignificance",0.0);
  desc.add<double>("MaxLxySignificance",0.0);
  desc.add<double>("MaxNormalisedChi2",10.0);
  desc.add<double>("MinVtxProbability",0.0);
  desc.add<double>("MinCosinePointingAngle",-2.0);
  desc.add<int>("triggerTypeDaughters",0);
  desc.add<edm::InputTag>("DisplacedVertexTag",edm::InputTag("hltDisplacedtktktkVtx"));
  desc.add<edm::InputTag>("BeamSpotTag",edm::InputTag("hltOnlineBeamSpot"));
  desc.add<edm::InputTag>("TrackTag",edm::InputTag("hltL3MuonCandidates"));
  descriptions.add("hltDisplacedtktktkFilter", desc);
}

// ------------ method called once each job just before starting event loop  ------------
void HLTDisplacedtktktkFilter::beginJob()
{

}

// ------------ method called once each job just after ending the event loop  ------------
void HLTDisplacedtktktkFilter::endJob()
{
 	
}

// ------------ method called on each new Event  ------------
bool HLTDisplacedtktktkFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{


  // get beam spot
  reco::BeamSpot vertexBeamSpot;
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByToken(beamSpotToken_,recoBeamSpotHandle);
  vertexBeamSpot = *recoBeamSpotHandle;


  // get displaced vertices
  reco::VertexCollection displacedVertexColl;
  edm::Handle<reco::VertexCollection> displacedVertexCollHandle;
  bool foundVertexColl = iEvent.getByToken(DisplacedVertexToken_, displacedVertexCollHandle);
  if(foundVertexColl) displacedVertexColl = *displacedVertexCollHandle;


  // get track collection
  edm::Handle<reco::RecoChargedCandidateCollection> trackcands;
  iEvent.getByToken(TrackToken_,trackcands);

  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.


  // Ref to Candidate object to be recorded in filter object
  reco::RecoChargedCandidateRef ref1;
  reco::RecoChargedCandidateRef ref2;
  reco::RecoChargedCandidateRef ref3;

  if (saveTags()) filterproduct.addCollectionTag(TrackTag_);

  bool triggered = false;

  // loop over vertex collection
  for(auto displacedVertex : displacedVertexColl){
    // check if the vertex actually consists of exactly three track tracks, reject the event if not
    if(displacedVertex.tracksSize() != 3){
      edm::LogError("HLTDisplacedtktktkFilter") << "HLTDisplacedtktktkFilter: ERROR: the tktktk vertex must have exactly three tracks by definition. It now has n tracks = "<< displacedVertex.tracksSize() << std::endl;
      return false;
    } 

    float normChi2 = displacedVertex.normalizedChi2();
    if (normChi2 > maxNormalisedChi2_) continue;

    double vtxProb = 0.0;
    if( (displacedVertex.chi2()>=0.0) && (displacedVertex.ndof()>0) ) vtxProb = TMath::Prob(displacedVertex.chi2(), displacedVertex.ndof() );
    if (vtxProb < minVtxProbability_) continue;

    // get the three tracks from the vertex
    auto trackIt =  displacedVertex.tracks_begin();
    reco::TrackRef vertextkRef1 =  (*trackIt).castTo<reco::TrackRef>();
    // the second one
    trackIt++;
    reco::TrackRef vertextkRef2 =  (*trackIt).castTo<reco::TrackRef>();
    // the third one
    trackIt++;
    reco::TrackRef vertextkRef3 =  (*trackIt).castTo<reco::TrackRef>();

    // first find these three tracks in the track collection
    reco::RecoChargedCandidateCollection::const_iterator cand1;
    reco::RecoChargedCandidateCollection::const_iterator cand2;	
    reco::RecoChargedCandidateCollection::const_iterator cand3;	

    int iFoundRefs = 0;
    for (auto cand=trackcands->begin(); cand!=trackcands->end(); cand++) {
      reco::TrackRef tkRef = cand->get<reco::TrackRef>();
      if(tkRef == vertextkRef1) {cand1 = cand; iFoundRefs++;}
      if(tkRef == vertextkRef2) {cand2 = cand; iFoundRefs++;}
      if(tkRef == vertextkRef3) {cand3 = cand; iFoundRefs++;}
    }
	  
    if(iFoundRefs < 3){
      edm::LogError("HLTDisplacedtktktkFilter") << "HLTDisplacedtktktkFilter: ERROR: the tracks matched with the tktktk vertex tracks should be at least three by definition."  << std::endl;
      return false;
    } 	
    // calculate three-track transverse momentum
    math::XYZVector pperp(cand1->px() + cand2->px() + cand3->px(),
                          cand1->py() + cand2->py() + cand3->py(),
                          0.);

    const reco::Vertex::Point& vpoint=displacedVertex.position();
    //translate to global point, should be improved
    GlobalPoint secondaryVertex (vpoint.x(), vpoint.y(), vpoint.z());

    reco::Vertex::Error verr = displacedVertex.error();
    // translate to global error, should be improved
    GlobalError err(verr.At(0,0), verr.At(1,0), verr.At(1,1), verr.At(2,0), verr.At(2,1), verr.At(2,2) );

    GlobalPoint displacementFromBeamspot( -1*((vertexBeamSpot.x0() -  secondaryVertex.x()) +  (secondaryVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dxdz()),
                                          -1*((vertexBeamSpot.y0() - secondaryVertex.y())+  (secondaryVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dydz()), 0);

    float lxy = displacementFromBeamspot.perp();
    float lxyerr = sqrt(err.rerr(displacementFromBeamspot));


    //calculate the angle between the decay length and the tktk momentum
    reco::Vertex::Point vperp(displacementFromBeamspot.x(),displacementFromBeamspot.y(),0.);

    float cosAlpha = vperp.Dot(pperp)/(vperp.R()*pperp.R());

    // check thresholds
    if (cosAlpha < minCosinePointingAngle_) continue;
    if (minLxySignificance_ > 0. && lxy/lxyerr < minLxySignificance_) continue;
    if (maxLxySignificance_ > 0. && lxy/lxyerr > maxLxySignificance_) continue;
    triggered = true;

    // now add the tracks that passed to the filter object
	
    ref1=reco::RecoChargedCandidateRef( edm::Ref<reco::RecoChargedCandidateCollection> 	(trackcands,distance(trackcands->begin(), cand1)));
    filterproduct.addObject(triggerTypeDaughters_,ref1);
	
    ref2=reco::RecoChargedCandidateRef( edm::Ref<reco::RecoChargedCandidateCollection> (trackcands,distance(trackcands->begin(),cand2 )));
    filterproduct.addObject(triggerTypeDaughters_,ref2);

    ref3=reco::RecoChargedCandidateRef( edm::Ref<reco::RecoChargedCandidateCollection> (trackcands,distance(trackcands->begin(),cand3 )));
    filterproduct.addObject(triggerTypeDaughters_,ref3);
  }

  LogDebug("HLTDisplacedtktktkFilter") << " >>>>> Result of HLTDisplacedtktktkFilter is "<< triggered <<std::endl;

  return triggered;
}
