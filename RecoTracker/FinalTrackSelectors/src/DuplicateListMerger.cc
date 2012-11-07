#include "RecoTracker/FinalTrackSelectors/interface/DuplicateListMerger.h"

#include "FWCore/Framework/interface/Event.h"

using reco::modules::DuplicateListMerger;

DuplicateListMerger::DuplicateListMerger(const edm::ParameterSet& iPara) 
{
  diffHitsCut_ = 9999;
  if(iPara.exists("diffHitsCut"))diffHitsCut_ = iPara.getParameter<int>("diffHitsCut");
  if(iPara.exists("mergedSource"))mergedTrackSource_ = iPara.getParameter<edm::InputTag>("mergedSource");
  if(iPara.exists("originalSource"))originalTrackSource_ = iPara.getParameter<edm::InputTag>("originalSource");
  if(iPara.exists("candidateSource"))candidateSource_ = iPara.getParameter<edm::InputTag>("candidateSource");

  produces<std::vector<reco::Track> >();


}

DuplicateListMerger::~DuplicateListMerger()
{

  /* no op */

}

void DuplicateListMerger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<edm::View<reco::Track> > originalHandle;
  iEvent.getByLabel(originalTrackSource_,originalHandle);
  edm::Handle<edm::View<reco::Track> > mergedHandle;
  iEvent.getByLabel(mergedTrackSource_,mergedHandle);

  edm::Handle<edm::View<DuplicateRecord> > candidateHandle;
  iEvent.getByLabel(candidateSource_,candidateHandle);

  std::auto_ptr<std::vector<reco::Track> > out_generalTracks(new std::vector<reco::Track>());

  //match new tracks to their candidates
  std::vector<std::pair<int,DuplicateRecord> > matches;
  for(int i = 0; i < (int)candidateHandle->size(); i++){
    DuplicateRecord duplicateRecord = candidateHandle->at(i);
    int matchTrack = matchCandidateToTrack(duplicateRecord.first,mergedHandle);
    if(matchTrack < 0)continue;
    unsigned int dHits = (duplicateRecord.first.recHits().second - duplicateRecord.first.recHits().first) - mergedHandle->at(matchTrack).recHitsSize();
    if(dHits > diffHitsCut_)continue;
    matches.push_back(std::pair<int,DuplicateRecord>(matchTrack,duplicateRecord));
  }

  //check for candidates/tracks that share merged tracks, select minimum chi2, remove the rest
  std::vector<std::pair<int,DuplicateRecord> >::iterator matchIter0 = matches.begin();
  std::vector<std::pair<int,DuplicateRecord> >::iterator matchIter1;
  while(matchIter0 != matches.end()){
    double nchi2 = (mergedHandle->at((*matchIter0).first)).normalizedChi2();
    bool advance = true;
    for(matchIter1 = matchIter0+1; matchIter1 != matches.end(); matchIter1++){
      if((*matchIter1).second.second.first.seedRef() == (*matchIter0).second.second.first.seedRef() || (*matchIter1).second.second.first.seedRef() == (*matchIter0).second.second.second.seedRef() || (*matchIter1).second.second.second.seedRef() == (*matchIter0).second.second.first.seedRef() || (*matchIter1).second.second.second.seedRef() == (*matchIter0).second.second.second.seedRef()){
	double nchi2_1 = (mergedHandle->at((*matchIter1).first)).normalizedChi2();
	advance = false;
	if(nchi2_1 < nchi2){
	  matches.erase(matchIter0);
	}else{
	  matches.erase(matchIter1);
	}
	break;
      }
    }
    if(advance)matchIter0++;
  }


  //add the good merged tracks to the output list, remove input tracks
  std::vector<reco::Track> inputTracks;
  for(matchIter0 = matches.begin(); matchIter0 != matches.end(); matchIter0++){
    out_generalTracks->push_back(mergedHandle->at((*matchIter0).first));
    inputTracks.push_back((*matchIter0).second.second.first);
    inputTracks.push_back((*matchIter0).second.second.second);
  }
  for(int i = 0; i < (int)originalHandle->size(); i++){
    bool good = true;
    reco::Track origTrack = originalHandle->at(i);
    for(int j = 0; j < (int)inputTracks.size() && good; j++){
      reco::Track inputTrack = inputTracks[j];
      if(origTrack.seedRef() != inputTrack.seedRef())continue;
      if(origTrack.charge() != inputTrack.charge())continue;
      if(origTrack.momentum() != inputTrack.momentum())continue;
      if(origTrack.referencePoint() != inputTrack.referencePoint())continue;
      good = false;
    }
    if(good)out_generalTracks->push_back(origTrack);
  }


  iEvent.put(out_generalTracks);
}

//------------------------------------------------------------
//------------------------------------------------------------
//------------------------------------------------------------
//------------------------------------------------------------
int DuplicateListMerger::matchCandidateToTrack(TrackCandidate candidate, edm::Handle<edm::View<reco::Track> > tracks){
  int track = -1;
  std::vector<int> rawIds;
  RecHitContainer::const_iterator candBegin = candidate.recHits().first;
  RecHitContainer::const_iterator candEnd = candidate.recHits().second;
  for(; candBegin != candEnd; candBegin++){
    rawIds.push_back((*(candBegin)).rawId());
  }
 

  for(int i = 0; i < (int)tracks->size() && track < 0;i++){
    if((tracks->at(i)).seedRef() != candidate.seedRef())continue;
    int match = 0;
    trackingRecHit_iterator trackRecBegin = tracks->at(i).recHitsBegin();
    trackingRecHit_iterator trackRecEnd = tracks->at(i).recHitsEnd();
    for(;trackRecBegin != trackRecEnd; trackRecBegin++){
      if(std::find(rawIds.begin(),rawIds.end(),(*(trackRecBegin)).get()->rawId()) != rawIds.end())match++;
    }
    if(match != (int)tracks->at(i).recHitsSize())continue;
    track = i;
  }

  return track;
}
//------------------------------------------------------------
//------------------------------------------------------------
//------------------------------------------------------------
//------------------------------------------------------------
