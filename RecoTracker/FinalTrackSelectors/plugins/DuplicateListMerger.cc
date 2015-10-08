/** \class DuplicateListMerger
 * 
 * merges list of merge duplicate tracks with its parent list
 *
 * \author Matthew Walker
 */


#include "RecoTracker/FinalTrackSelectors/interface/TrackCollectionCloner.h"


#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <map>
#include<memory>

#include "TMVA/Reader.h"

#include "trackAlgoPriorityOrder.h"

using namespace reco;


class dso_hidden DuplicateListMerger final : public edm::stream::EDProducer<> {
 public:
  /// constructor
  explicit DuplicateListMerger(const edm::ParameterSet& iPara);
  /// destructor
  virtual ~DuplicateListMerger();
  
  /// alias for container of candidate and input tracks
  using CandidateToDuplicate = std::vector<std::pair<int, int>>;
  
  using RecHitContainer = edm::OwnVector<TrackingRecHit>;
  
  using MVACollection = std::vector<float>;
  using QualityMaskCollection = std::vector<unsigned char>;
 
  
 protected:
  /// produce one event
  void produce( edm::Event &, const edm::EventSetup &) override;
  
 private:
  int matchCandidateToTrack(TrackCandidate const &, reco::TrackCollection const &);
  
  

  TrackCollectionCloner collectionCloner;
  TrackCollectionCloner::Tokens mergedTrackSource_;
  TrackCollectionCloner::Tokens originalTrackSource_;


  edm::EDGetTokenT<CandidateToDuplicate> candidateComponents_;
  edm::EDGetTokenT<std::vector<TrackCandidate>> candidateSource_;
	 
  edm::EDGetTokenT<MVACollection> originalMVAValsToken_;
  edm::EDGetTokenT<MVACollection> mergedMVAValsToken_;

  reco::TrackBase::TrackQuality qualityToSet_;
  unsigned int diffHitsCut_;
  float minTrkProbCut_;
 

};
 

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/ClusterRemovalRefSetter.h"

#include "FWCore/Framework/interface/Event.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

#include<tuple>
#include<array>
#include "CommonTools/Utils/interface/DynArray.h"

DuplicateListMerger::DuplicateListMerger(const edm::ParameterSet& iPara) :
  collectionCloner(*this, iPara, true),
  mergedTrackSource_(iPara.getParameter<edm::InputTag>("mergedSource"),consumesCollector()),
  originalTrackSource_(iPara.getParameter<edm::InputTag>("originalSource"),consumesCollector())
{
  
  diffHitsCut_ = iPara.getParameter<int>("diffHitsCut");
  minTrkProbCut_ = iPara.getParameter<double>("minTrkProbCut");
  candidateSource_ = consumes<std::vector<TrackCandidate> >(iPara.getParameter<edm::InputTag>("candidateSource"));
  candidateComponents_ = consumes<CandidateToDuplicate>(iPara.getParameter<edm::InputTag>("candidateComponents"));

  mergedMVAValsToken_ = consumes<MVACollection>(iPara.getParameter<edm::InputTag>("mergedMVAVals"));
  originalMVAValsToken_ = consumes<MVACollection>(iPara.getParameter<edm::InputTag>("originalMVAVals"));

  

  produces<MVACollection>("MVAValues");
  produces<QualityMaskCollection>("QualityMasks");


}

DuplicateListMerger::~DuplicateListMerger()
{

  /* no op */

}

void DuplicateListMerger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  TrackCollectionCloner::Producer producer(iEvent, collectionCloner);

  
 
  auto const & originals = originalTrackSource_.tracks(iEvent);
  auto const & merged = mergedTrackSource_.tracks(iEvent);
  auto const & candIndices = mergedTrackSource_.indicesInput(iEvent);
  
  edm::Handle<std::vector<TrackCandidate>> candidateH;
  iEvent.getByToken(candidateSource_,candidateH);
  auto const & candidates = *candidateH;

  edm::Handle<CandidateToDuplicate>  candidateComponentsH;
  iEvent.getByToken(candidateComponents_,candidateComponentsH);
  auto const & candidateComponents = *candidateComponentsH;


  
  edm::Handle<MVACollection> originalMVAStore;
  edm::Handle<MVACollection> mergedMVAStore;

  iEvent.getByToken(originalMVAValsToken_,originalMVAStore);
  iEvent.getByToken(mergedMVAValsToken_,mergedMVAStore);

  MVACollection mvaVec;


  auto mergedMVA = *mergedMVAStore;

  //match new tracks to their candidates
  std::vector<std::array<int,3>> matches;
  for(int i = 0; i < (int)merged.size(); ++i) {
    auto cInd = candIndices[i];
    auto const & cand = candidates[cInd];
    const reco::Track& matchedTrack = merged[i];

    if (mergedMVA[i]< -0.7f) continue;

    if( ChiSquaredProbability(matchedTrack.chi2(),matchedTrack.ndof()) < minTrkProbCut_)continue;
    unsigned int dHits = (cand.recHits().second - cand.recHits().first) - matchedTrack.recHitsSize();
    if(dHits > diffHitsCut_)continue;
    matches.push_back(std::array<int,3>{{i,candidateComponents[cInd].first,candidateComponents[cInd].second}});
  }

  //check for candidates/tracks that share merged tracks, select minimum chi2, remove the rest
  if (matches.size()>1) 
    for ( auto matchIter0 = matches.begin(); matchIter0 != matches.end()-1; ++matchIter0) {
      if ( (*matchIter0)[0]<0) continue;
      auto nchi2 = merged[(*matchIter0)[0]].normalizedChi2();
      for( auto matchIter1 = matchIter0+1; matchIter1 != matches.end(); ++matchIter1){
	if ( (*matchIter1)[0]<0) continue;
	if ( (*matchIter0)[1]==(*matchIter1)[1]
	     || (*matchIter0)[1]==(*matchIter1)[2]
	     || (*matchIter0)[2]==(*matchIter1)[1]
	     || (*matchIter0)[2]==(*matchIter1)[2]
	     ) {
	  auto nchi2_1 = merged[(*matchIter1)[0]].normalizedChi2();
	  if(nchi2_1 < nchi2){
	    (*matchIter0)[0] = -1;
	    break;
	  }else{
	    (*matchIter1)[0] = -1;
	  }
	}
      }
    }


  // products
  auto pmvas = std::make_unique<MVACollection>();
  auto pquals = std::make_unique<QualityMaskCollection>();
  

  //add the good merged tracks to the output list, remove input tracks
  std::vector<int> inputTracks;


  std::vector<unsigned int> selId;
  auto ntotTk = matches.size();
  // declareDynArray(reco::TrackBase::TrackAlgorithm, ntotTk, algo);
  declareDynArray(reco::TrackBase::TrackAlgorithm, ntotTk, oriAlgo);
  declareDynArray(reco::TrackBase::AlgoMask, ntotTk,  algoMask);

  auto nsel=0U;
  for(auto matchIter0 = matches.begin(); matchIter0 != matches.end(); matchIter0++){
    if ( (*matchIter0)[0]<0) continue;
    selId.push_back((*matchIter0)[0]);

    pmvas->push_back(mergedMVA[(*matchIter0)[0]]);
     
    const reco::Track& inTrk1 = originals[(*matchIter0)[1]];
    const reco::Track& inTrk2 = originals[(*matchIter0)[2]];
    oriAlgo[nsel] = std::min(inTrk1.algo(),inTrk2.algo(),
			     [](reco::TrackBase::TrackAlgorithm a, reco::TrackBase::TrackAlgorithm b) {
			       return trackAlgoPriorityOrder[a] < trackAlgoPriorityOrder[b];
			     });


    algoMask[nsel] = inTrk1.algoMask() | inTrk2.algoMask();
    
    pquals->push_back((inTrk1.qualityMask() | inTrk2.qualityMask()));
    
    inputTracks.push_back((*matchIter0)[1]);
    inputTracks.push_back((*matchIter0)[2]);

 
    ++nsel;
    
  }

  
  producer(mergedTrackSource_,selId);
  assert(producer.selTracks_->size()==pquals->size());
  
  for (auto isel=0U;isel<nsel;++isel) {
    auto & otk = (*producer.selTracks_)[isel];
    otk.setQualityMask((*pquals)[isel]);
    otk.setAlgorithm(oriAlgo[isel]);
    //    otk.setOriginalAlgorithm(oriAlgo[isel]);
    otk.setAlgoMask(algoMask[isel]);
  }

  
  selId.clear();
  for(int i = 0; i < (int)originals.size(); i++){
    const reco::Track& origTrack = originals[i];
    if (std::find(inputTracks.begin(),inputTracks.end(),i) != inputTracks.end()) continue;
    selId.push_back(i);
    pmvas->push_back((*originalMVAStore)[i]);
    pquals->push_back(origTrack.qualityMask());
  }

  producer(originalTrackSource_,selId);
  assert(producer.selTracks_->size()==pquals->size());

  iEvent.put(std::move(pmvas),"MVAValues");
  iEvent.put(std::move(pquals),"QualityMasks");


}

//------------------------------------------------------------
//------------------------------------------------------------
//------------------------------------------------------------
//------------------------------------------------------------
int DuplicateListMerger::matchCandidateToTrack(TrackCandidate  const & candidate, reco::TrackCollection const & tracks){
  int track = -1;
  std::vector<int> rawIds;
  RecHitContainer::const_iterator candBegin = candidate.recHits().first;
  RecHitContainer::const_iterator candEnd = candidate.recHits().second;
  for(; candBegin != candEnd; candBegin++){
    rawIds.push_back((*(candBegin)).rawId());
  }
 

  for(int i = 0; i < (int)tracks.size() && track < 0;i++){
    if( (tracks)[i].seedRef() != candidate.seedRef())continue;
    int match = 0;
    trackingRecHit_iterator trackRecBegin = (tracks)[i].recHitsBegin();
    trackingRecHit_iterator trackRecEnd = (tracks)[i].recHitsEnd();
    for(;trackRecBegin != trackRecEnd; trackRecBegin++){
      if(std::find(rawIds.begin(),rawIds.end(),(*(trackRecBegin))->rawId()) != rawIds.end()) match++;
    }
    if(match != (int)( (tracks)[i].recHitsSize() ) ) continue;
    track = i;
  }

  return track;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(DuplicateListMerger);
