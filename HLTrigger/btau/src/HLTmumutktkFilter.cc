#include <algorithm>
#include <cmath>
#include <vector>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "HLTrigger/btau/src/HLTmumutktkFilter.h"
#include "TMath.h"

using namespace edm;
using namespace reco;
using namespace std;
using namespace trigger;

// ----------------------------------------------------------------------
HLTmumutktkFilter::HLTmumutktkFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
  muCandTag_  (iConfig.getParameter<edm::InputTag>("MuonTag")),
  muCandToken_(consumes<reco::RecoChargedCandidateCollection>(muCandTag_)),
  trkCandTag_  (iConfig.getParameter<edm::InputTag>("TrackTag")),
  trkCandToken_(consumes<reco::RecoChargedCandidateCollection>(trkCandTag_)),
  MuMuTkVertexTag_  (iConfig.getParameter<edm::InputTag>("MuMuTkVertexTag")),
  MuMuTkVertexToken_(consumes<reco::VertexCollection>(MuMuTkVertexTag_)),
  beamSpotTag_ (iConfig.getParameter<edm::InputTag> ("BeamSpotTag")),
  beamSpotToken_(consumes<reco::BeamSpot>(beamSpotTag_)),
  maxEta_(iConfig.getParameter<double>("MaxEta")),
  minPt_(iConfig.getParameter<double>("MinPt")),
  maxNormalisedChi2_(iConfig.getParameter<double>("MaxNormalisedChi2")),
  minVtxProbability_(iConfig.getParameter<double>("MinVtxProbability")),
  minLxySignificance_(iConfig.getParameter<double>("MinLxySignificance")),
  minCosinePointingAngle_(iConfig.getParameter<double>("MinCosinePointingAngle"))
{
}

// ----------------------------------------------------------------------
HLTmumutktkFilter::~HLTmumutktkFilter() {}

void
HLTmumutktkFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<double>("MaxEta",2.5);
  desc.add<double>("MinPt" ,0.0);
  desc.add<double>("MaxNormalisedChi2" ,10.0);
  desc.add<double>("MinVtxProbability" , 0.0);
  desc.add<double>("MinLxySignificance",3.0);
  desc.add<double>("MinCosinePointingAngle",0.9);
  desc.add<edm::InputTag>("MuonTag",edm::InputTag("hltL3MuonCandidates"));
  desc.add<edm::InputTag>("TrackTag",edm::InputTag("hltMumukAllConeTracks"));
  desc.add<edm::InputTag>("MuMuTkVertexTag",edm::InputTag("hltDisplacedmumuVtxProducerDoubleMu4Jpsi"));
  desc.add<edm::InputTag>("BeamSpotTag",edm::InputTag("hltOnineBeamSpot"));
  descriptions.add("HLTmumutktkFilter",desc);
}

// ----------------------------------------------------------------------
bool HLTmumutktkFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const {

  //get the beamspot position
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByToken(beamSpotToken_,recoBeamSpotHandle);
  const reco::BeamSpot& vertexBeamSpot = *recoBeamSpotHandle;

  // get vertices
  reco::VertexCollection displacedVertexColl;
  edm::Handle<reco::VertexCollection> displacedVertexCollHandle;
  bool foundVertexColl = iEvent.getByToken(MuMuTkVertexToken_, displacedVertexCollHandle);
  if(foundVertexColl) displacedVertexColl = *displacedVertexCollHandle;

  // get muon collection
  Handle<RecoChargedCandidateCollection> mucands;
  iEvent.getByToken(muCandToken_,mucands);

  // get track candidates around displaced muons
  Handle<RecoChargedCandidateCollection> trkcands;
  iEvent.getByToken(trkCandToken_,trkcands);

  // Ref to Candidate object to be recorded in filter object
  RecoChargedCandidateRef refMu1 ;
  RecoChargedCandidateRef refMu2 ;
  RecoChargedCandidateRef refTrk1;    
  RecoChargedCandidateRef refTrk2;    
    
  if (saveTags()) {
    filterproduct.addCollectionTag(muCandTag_);
    filterproduct.addCollectionTag(trkCandTag_);
  }

  bool triggered = false;

  // loop over vertex collection
  reco::VertexCollection::iterator it;
  for(it = displacedVertexColl.begin(); it!= displacedVertexColl.end(); it++){
    reco::Vertex displacedVertex = *it;

    // check if the vertex actually consists of exactly two muon + 1 track, throw exception if not
    if(displacedVertex.tracksSize() != 4) throw cms::Exception("BadLogic") << "HLTmumutktkFilter: ERROR: the Jpsi+trk vertex must have " 
                                                                           << "exactly two muons + 2 trk by definition. It now has n trakcs = "
                                                                           << displacedVertex.tracksSize() << std::endl;

    float normChi2 = displacedVertex.normalizedChi2();
    if (normChi2 > maxNormalisedChi2_) continue;

    double vtxProb = 0.0;
    if ((displacedVertex.chi2()>=0.0) && (displacedVertex.ndof()>0) ) 
      vtxProb = TMath::Prob(displacedVertex.chi2(), displacedVertex.ndof() );
    if (vtxProb < minVtxProbability_) continue;

    // get the four tracks from the vertex
    std::vector <reco::TrackRef> vertexTrksRefVec;
    reco::Vertex::trackRef_iterator trackIt =  displacedVertex.tracks_begin();
    for (trackIt = displacedVertex.tracks_begin(); trackIt != displacedVertex.tracks_end(); trackIt++){
        vertexTrksRefVec.push_back((*trackIt).castTo<reco::TrackRef>());
    }

    // first find the tracks in the input muon/track collection
    std::vector<reco::RecoChargedCandidateCollection::const_iterator> mucandVec;
    std::vector<reco::RecoChargedCandidateCollection::const_iterator> trkcandVec;
    for (reco::RecoChargedCandidateCollection::const_iterator cand=mucands->begin(); cand!=mucands->end(); cand++) {
      reco::TrackRef tkRef = cand->get<reco::TrackRef>();
      for (unsigned int iVec=0; iVec < vertexTrksRefVec.size();iVec++){
        if (tkRef == vertexTrksRefVec.at(iVec)) {mucandVec.push_back(cand); break;}
      }
    }
    if(mucandVec.size()!= 2) throw cms::Exception("BadLogic") << "HLTmumutktkFilterr: ERROR: the vertex must have "
                                                              << " exactly two muons by definition."  << std::endl;

    for (reco::RecoChargedCandidateCollection::const_iterator cand=trkcands->begin(); cand!=trkcands->end(); cand++) {
      reco::TrackRef tkRef = cand->get<reco::TrackRef>();
      for (unsigned int iVec = 0; iVec < vertexTrksRefVec.size();iVec++){
        if (tkRef == vertexTrksRefVec.at(iVec)) {trkcandVec.push_back(cand); break;}
      }
    }
    if(trkcandVec.size()!= 2 ) throw cms::Exception("BadLogic") << "HLTmumutktkFilterr: ERROR: the vertex must have "
                                                                << " exactly two tracks by definition."  << std::endl;

    // calculate four-track transverse momentum
    math::XYZVector pperp(mucandVec.at(0)->px() + mucandVec.at(1)->px() + trkcandVec.at(0)->px() + trkcandVec.at(1)->px(),
                          mucandVec.at(0)->py() + mucandVec.at(1)->py() + trkcandVec.at(0)->px() + trkcandVec.at(1)->py(),
                          0.);
                          
    // get vertex position and error to calculate the decay length significance
    reco::Vertex::Point vpoint=displacedVertex.position();
    reco::Vertex::Error verr = displacedVertex.error();
    GlobalPoint secondaryVertex (vpoint.x(), vpoint.y(), vpoint.z());
    GlobalError err(verr.At(0,0), verr.At(1,0), verr.At(1,1), verr.At(2,0), verr.At(2,1), verr.At(2,2) );

    GlobalPoint displacementFromBeamspot( -1*((vertexBeamSpot.x0() - secondaryVertex.x()) + (secondaryVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dxdz()), 
                                          -1*((vertexBeamSpot.y0() - secondaryVertex.y()) + (secondaryVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dydz()), 
                                           0 );
    float lxy     = displacementFromBeamspot.perp();
    float lxyerr  = sqrt(err.rerr(displacementFromBeamspot));
    float lxysign = 0;
    if (lxyerr != 0)  lxysign = lxy/lxyerr;

    //calculate the angle between the decay length and the four-track momentum
    Vertex::Point vperp(displacementFromBeamspot.x(),displacementFromBeamspot.y(),0.);
    float cosAlpha = vperp.Dot(pperp)/(vperp.Rho()*pperp.Rho());

    if (pperp.Rho() < minPt_                 ) continue;
    if (lxysign     < minLxySignificance_    ) continue;
    if (cosAlpha    < minCosinePointingAngle_) continue;
    triggered = true;
          
    refMu1=RecoChargedCandidateRef( Ref<RecoChargedCandidateCollection> (mucands,distance(mucands->begin(), mucandVec.at(0))));
    filterproduct.addObject(TriggerMuon,refMu1);
    refMu2=RecoChargedCandidateRef( Ref<RecoChargedCandidateCollection> (mucands,distance(mucands->begin(), mucandVec.at(1))));
    filterproduct.addObject(TriggerMuon,refMu2);
    refTrk1=RecoChargedCandidateRef( Ref<RecoChargedCandidateCollection> (trkcands,distance(trkcands->begin(),trkcandVec.at(0))));
    filterproduct.addObject(TriggerTrack,refTrk1);
    refTrk2=RecoChargedCandidateRef( Ref<RecoChargedCandidateCollection> (trkcands,distance(trkcands->begin(),trkcandVec.at(1))));
    filterproduct.addObject(TriggerTrack,refTrk2);
          
  }//end loop vertices

  LogDebug("HLTDisplacedMumuTrkFilter") << " >>>>> Result of HLTDisplacedMuMuTrkFilter is "<< triggered;
  return triggered;
}



bool HLTmumutktkFilter::triggerdByPreviousLevel(const reco::RecoChargedCandidateRef & candref, const std::vector<reco::RecoChargedCandidateRef>& vcands){
  unsigned int i=0;
  unsigned int i_max=vcands.size();
  for (;i!=i_max;++i){
    if (candref == vcands[i]) return true;
  }
  return false;
}