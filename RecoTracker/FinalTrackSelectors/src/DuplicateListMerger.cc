#include "RecoTracker/FinalTrackSelectors/interface/DuplicateListMerger.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoTracker/TrackProducer/interface/ClusterRemovalRefSetter.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "FWCore/Framework/interface/Event.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

using reco::modules::DuplicateListMerger;

DuplicateListMerger::DuplicateListMerger(const edm::ParameterSet& iPara) 
{
  diffHitsCut_ = 9999;
  minTrkProbCut_ = 0.0;
  if(iPara.exists("diffHitsCut"))diffHitsCut_ = iPara.getParameter<int>("diffHitsCut");
  if(iPara.exists("minTrkProbCut"))minTrkProbCut_ = iPara.getParameter<double>("minTrkProbCut");
  if(iPara.exists("mergedSource"))mergedTrackSource_ = iPara.getParameter<edm::InputTag>("mergedSource");
  if(iPara.exists("originalSource"))originalTrackSource_ = iPara.getParameter<edm::InputTag>("originalSource");
  if(iPara.exists("candidateSource"))candidateSource_ = iPara.getParameter<edm::InputTag>("candidateSource");


  if(iPara.exists("mergedMVAVals")){
    mergedMVAVals_ = iPara.getParameter<edm::InputTag>("mergedMVAVals");
  }else{
    mergedMVAVals_ = edm::InputTag(mergedTrackSource_.label(),"MVAVals");
  }
  if(iPara.exists("originalMVAVals")){
    originalMVAVals_ = iPara.getParameter<edm::InputTag>("originalMVAVals");
  }else{
    originalMVAVals_ = edm::InputTag(originalTrackSource_.label(),"MVAVals");
  }

  copyExtras_ = iPara.getUntrackedParameter<bool>("copyExtras",true);
  qualityToSet_ = reco::TrackBase::undefQuality;
  if (iPara.exists("newQuality")) {
    std::string qualityStr = iPara.getParameter<std::string>("newQuality");
    if (qualityStr != "") {
      qualityToSet_ = reco::TrackBase::qualityByName(qualityStr);
    }
  }

  produces<std::vector<reco::Track> >();
  produces< std::vector<Trajectory> >();
  produces< TrajTrackAssociationCollection >();

  produces<edm::ValueMap<float> >("MVAVals");

  makeReKeyedSeeds_ = iPara.getUntrackedParameter<bool>("makeReKeyedSeeds",false);
  if (makeReKeyedSeeds_){
    copyExtras_=true;
    produces<TrajectorySeedCollection>();
  }
  if(copyExtras_){
    produces<reco::TrackExtraCollection>();
    produces<TrackingRecHitCollection>();
  }

}

DuplicateListMerger::~DuplicateListMerger()
{

  /* no op */

}

void DuplicateListMerger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<reco::TrackCollection > originalHandle;
  iEvent.getByLabel(originalTrackSource_,originalHandle);
  edm::Handle<reco::TrackCollection > mergedHandle;
  iEvent.getByLabel(mergedTrackSource_,mergedHandle);

  const reco::TrackCollection& mergedTracks(*mergedHandle);
  reco::TrackRefProd originalTrackRefs(originalHandle);
  reco::TrackRefProd mergedTrackRefs(mergedHandle);

  edm::Handle< std::vector<Trajectory> >  mergedTrajHandle;
  iEvent.getByLabel(mergedTrackSource_,mergedTrajHandle);
  edm::Handle< TrajTrackAssociationCollection >  mergedTrajTrackHandle;
  iEvent.getByLabel(mergedTrackSource_,mergedTrajTrackHandle);

  edm::Handle< std::vector<Trajectory> >  originalTrajHandle;
  iEvent.getByLabel(originalTrackSource_,originalTrajHandle);
  edm::Handle< TrajTrackAssociationCollection >  originalTrajTrackHandle;
  iEvent.getByLabel(originalTrackSource_,originalTrajTrackHandle);

  edm::Handle<edm::View<DuplicateRecord> > candidateHandle;
  iEvent.getByLabel(candidateSource_,candidateHandle);

  std::auto_ptr<std::vector<reco::Track> > out_generalTracks(new std::vector<reco::Track>());
  out_generalTracks->reserve(originalHandle->size());
  reco::TrackRefProd refTrks = iEvent.getRefBeforePut<reco::TrackCollection>();
  std::auto_ptr< std::vector<Trajectory> > outputTrajs = std::auto_ptr< std::vector<Trajectory> >(new std::vector<Trajectory>());
  outputTrajs->reserve(originalTrajHandle->size()+mergedTrajHandle->size());
  edm::RefProd< std::vector<Trajectory> > refTrajs;
  std::auto_ptr< TrajTrackAssociationCollection >  outputTTAss = std::auto_ptr< TrajTrackAssociationCollection >(new TrajTrackAssociationCollection());
  //std::auto_ptr< TrajectorySeedCollection > outputSeeds

  std::auto_ptr<reco::TrackExtraCollection> outputTrkExtras;
  reco::TrackExtraRefProd refTrkExtras;
  std::auto_ptr<TrackingRecHitCollection> outputTrkHits;
  TrackingRecHitRefProd refTrkHits;
  std::auto_ptr<TrajectorySeedCollection> outputSeeds;
  edm::RefProd< TrajectorySeedCollection > refTrajSeeds;

  const int rSize = (int)originalHandle->size();
  edm::RefToBase<TrajectorySeed> seedsRefs[rSize];

  edm::Handle<edm::ValueMap<float> > originalMVAStore;
  edm::Handle<edm::ValueMap<float> > mergedMVAStore;

  iEvent.getByLabel(originalMVAVals_,originalMVAStore);
  iEvent.getByLabel(mergedMVAVals_,mergedMVAStore);

  std::auto_ptr<edm::ValueMap<float> > vmMVA = std::auto_ptr<edm::ValueMap<float> >(new edm::ValueMap<float>);
  edm::ValueMap<float>::Filler fillerMVA(*vmMVA);
  std::vector<float> mvaVec;


  if(copyExtras_){
    outputTrkExtras = std::auto_ptr<reco::TrackExtraCollection>(new reco::TrackExtraCollection);
    outputTrkExtras->reserve(originalHandle->size());
    refTrkExtras    = iEvent.getRefBeforePut<reco::TrackExtraCollection>();
    outputTrkHits   = std::auto_ptr<TrackingRecHitCollection>(new TrackingRecHitCollection);
    outputTrkHits->reserve(originalHandle->size()*25);
    refTrkHits      = iEvent.getRefBeforePut<TrackingRecHitCollection>();
    if (makeReKeyedSeeds_){
      outputSeeds = std::auto_ptr<TrajectorySeedCollection>(new TrajectorySeedCollection);
      outputSeeds->reserve(originalHandle->size());
      refTrajSeeds = iEvent.getRefBeforePut<TrajectorySeedCollection>();
    }
  }

  //match new tracks to their candidates
  std::vector<std::pair<int,const DuplicateRecord*> > matches;
  for(int i = 0; i < (int)candidateHandle->size(); i++){
    const DuplicateRecord& duplicateRecord = candidateHandle->at(i);
    int matchTrack = matchCandidateToTrack(duplicateRecord.first,mergedHandle);
    if(matchTrack < 0)continue;
    const reco::Track& matchedTrack(mergedTracks[matchTrack]);
    if( ChiSquaredProbability(matchedTrack.chi2(),matchedTrack.ndof()) < minTrkProbCut_)continue;
    unsigned int dHits = (duplicateRecord.first.recHits().second - duplicateRecord.first.recHits().first) - matchedTrack.recHitsSize();
    if(dHits > diffHitsCut_)continue;
    matches.push_back(std::pair<int,const DuplicateRecord*>(matchTrack,&duplicateRecord));
  }

  //check for candidates/tracks that share merged tracks, select minimum chi2, remove the rest
  std::vector<std::pair<int,const DuplicateRecord*> >::iterator matchIter0 = matches.begin();
  std::vector<std::pair<int,const DuplicateRecord*> >::iterator matchIter1;
  while(matchIter0 != matches.end()){
    double nchi2 = mergedTracks[matchIter0->first].normalizedChi2();
    bool advance = true;
    for(matchIter1 = matchIter0+1; matchIter1 != matches.end(); matchIter1++){
      const reco::Track& match0first = *(matchIter0->second->second.first.get());
      const reco::Track& match0second = *(matchIter0->second->second.second.get());
      const reco::Track& match1first = *(matchIter1->second->second.first.get());
      const reco::Track& match1second = *(matchIter1->second->second.second.get());
      if(match1first.seedRef() == match0first.seedRef() ||
	 match1first.seedRef() == match0second.seedRef() || 
	 match1second.seedRef() == match0first.seedRef() || 
	 match1second.seedRef() == match0second.seedRef()){
	double nchi2_1 = mergedTracks[matchIter1->first].normalizedChi2();
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

  refTrajs = iEvent.getRefBeforePut< std::vector<Trajectory> >();

  for(matchIter0 = matches.begin(); matchIter0 != matches.end(); matchIter0++){
    reco::TrackRef inTrkRef1 = matchIter0->second->second.first;
    reco::TrackRef inTrkRef2 = matchIter0->second->second.second;
    const reco::Track& inTrk1 = *(inTrkRef1.get());
    const reco::Track& inTrk2 = *(inTrkRef2.get());
    reco::TrackBase::TrackAlgorithm newTrkAlgo = std::min(inTrk1.algo(),inTrk2.algo());
    int combinedQualityMask = (inTrk1.qualityMask() | inTrk2.qualityMask());
    inputTracks.push_back(inTrk1);
    inputTracks.push_back(inTrk2);
    out_generalTracks->push_back(mergedTracks[matchIter0->first]);
    reco::TrackRef curTrackRef = reco::TrackRef(refTrks, out_generalTracks->size() - 1);
    float mergedMVA = (*mergedMVAStore)[reco::TrackRef(mergedTrackRefs,matchIter0->first)];
    mvaVec.push_back(mergedMVA);
    out_generalTracks->back().setAlgorithm(newTrkAlgo);
    out_generalTracks->back().setQualityMask(combinedQualityMask);
    out_generalTracks->back().setQuality(qualityToSet_);
    edm::RefToBase<TrajectorySeed> origSeedRef;
    if(copyExtras_){
      const reco::Track& track = mergedTracks[matchIter0->first];
      origSeedRef = track.seedRef();
      //creating a seed with rekeyed clusters if required
      if (makeReKeyedSeeds_){
	bool doRekeyOnThisSeed=false;
	
	edm::InputTag clusterRemovalInfos("");
	//grab on of the hits of the seed
	if (origSeedRef->nHits()!=0){
	  TrajectorySeed::const_iterator firstHit=origSeedRef->recHits().first;
	  const TrackingRecHit *hit = &*firstHit;
	  if (firstHit->isValid()){
	    edm::ProductID  pID=clusterProductB(hit);
	    // the cluster collection either produced a removalInfo or mot
	    //get the clusterremoval info from the provenance: will rekey if this is found
	    edm::Handle<reco::ClusterRemovalInfo> CRIh;
	    edm::Provenance prov=iEvent.getProvenance(pID);
	    clusterRemovalInfos=edm::InputTag(prov.moduleLabel(),
					      prov.productInstanceName(),
					      prov.processName());
	    doRekeyOnThisSeed=iEvent.getByLabel(clusterRemovalInfos,CRIh);
	  }//valid hit
	}//nhit!=0
	
	if (doRekeyOnThisSeed && !(clusterRemovalInfos==edm::InputTag(""))) {
	  ClusterRemovalRefSetter refSetter(iEvent,clusterRemovalInfos);
	  TrajectorySeed::recHitContainer  newRecHitContainer;
	  newRecHitContainer.reserve(origSeedRef->nHits());
	  TrajectorySeed::const_iterator iH=origSeedRef->recHits().first;
	  TrajectorySeed::const_iterator iH_end=origSeedRef->recHits().second;
	  for (;iH!=iH_end;++iH){
	    newRecHitContainer.push_back(*iH);
	    refSetter.reKey(&newRecHitContainer.back());
	  }
	  outputSeeds->push_back( TrajectorySeed( origSeedRef->startingState(),
						  newRecHitContainer,
						  origSeedRef->direction()));
	}
	//doRekeyOnThisSeed=true
	else{
	  //just copy the one we had before
	  outputSeeds->push_back( TrajectorySeed(*origSeedRef));
	  }
	edm::Ref<TrajectorySeedCollection> pureRef(refTrajSeeds, outputSeeds->size()-1);
	origSeedRef=edm::RefToBase<TrajectorySeed>( pureRef);
      }//creating a new seed and rekeying it rechit clusters.
      // Fill TrackExtra collection
      outputTrkExtras->push_back( reco::TrackExtra( 
						   track.outerPosition(), track.outerMomentum(), track.outerOk(),
						   track.innerPosition(), track.innerMomentum(), track.innerOk(),
						   track.outerStateCovariance(), track.outerDetId(),
						   track.innerStateCovariance(), track.innerDetId(),
						   track.seedDirection(), origSeedRef ) );
      seedsRefs[(*matchIter0).first]=origSeedRef;
      out_generalTracks->back().setExtra( reco::TrackExtraRef( refTrkExtras, outputTrkExtras->size() - 1) );
      reco::TrackExtra & tx = outputTrkExtras->back();
      tx.setResiduals(track.residuals());
      // fill TrackingRecHits
      unsigned nh1=track.recHitsSize();
      for ( unsigned ih=0; ih<nh1; ++ih ) { 
	  //const TrackingRecHit*hit=&((*(track->recHit(ih))));
	outputTrkHits->push_back( track.recHit(ih)->clone() );
	tx.add( TrackingRecHitRef( refTrkHits, outputTrkHits->size() - 1) );
      }
    }
    edm::Ref< std::vector<Trajectory> > trajRef(mergedTrajHandle, (*matchIter0).first);
    TrajTrackAssociationCollection::const_iterator match = mergedTrajTrackHandle->find(trajRef);
    if (match != mergedTrajTrackHandle->end()) {
	if(curTrackRef.isNonnull()){
	  outputTrajs->push_back( *trajRef );
	  if (copyExtras_ && makeReKeyedSeeds_)
	    outputTrajs->back().setSeedRef( origSeedRef );
	  outputTTAss->insert(edm::Ref< std::vector<Trajectory> >(refTrajs, outputTrajs->size() - 1),curTrackRef );
	}
    }
  }

  for(int i = 0; i < (int)originalHandle->size(); i++){
    bool good = true;
    const reco::Track& origTrack = originalHandle->at(i);
    for(int j = 0; j < (int)inputTracks.size() && good; j++){
      const reco::Track& inputTrack = inputTracks[j];
      if(origTrack.seedRef() != inputTrack.seedRef())continue;
      if(origTrack.charge() != inputTrack.charge())continue;
      if(origTrack.momentum() != inputTrack.momentum())continue;
      if(origTrack.referencePoint() != inputTrack.referencePoint())continue;
      good = false;
    }

    if(good){
      out_generalTracks->push_back(origTrack);
      reco::TrackRef curTrackRef = reco::TrackRef(refTrks, out_generalTracks->size() - 1);
      edm::RefToBase<TrajectorySeed> origSeedRef;
      reco::TrackRef origTrackRef = reco::TrackRef(originalHandle,i);
      mvaVec.push_back((*originalMVAStore)[origTrackRef]);
      //mvaVec.push_back((*originalMVAStore)[reco::TrackRef(originalTrackRefs,i)]);
      if(copyExtras_){
	const reco::Track& track = origTrack;
	origSeedRef = track.seedRef();
	//creating a seed with rekeyed clusters if required
	if (makeReKeyedSeeds_){
	  bool doRekeyOnThisSeed=false;
	  
	  edm::InputTag clusterRemovalInfos("");
	  //grab on of the hits of the seed
	  if (origSeedRef->nHits()!=0){
	    TrajectorySeed::const_iterator firstHit=origSeedRef->recHits().first;
	    const TrackingRecHit *hit = &*firstHit;
	    if (firstHit->isValid()){
	      edm::ProductID  pID=clusterProductB(hit);
	      // the cluster collection either produced a removalInfo or mot
	      //get the clusterremoval info from the provenance: will rekey if this is found
	      edm::Handle<reco::ClusterRemovalInfo> CRIh;
	      edm::Provenance prov=iEvent.getProvenance(pID);
	      clusterRemovalInfos=edm::InputTag(prov.moduleLabel(),
						prov.productInstanceName(),
						prov.processName());
	      doRekeyOnThisSeed=iEvent.getByLabel(clusterRemovalInfos,CRIh);
	    }//valid hit
	  }//nhit!=0
	  
	  if (doRekeyOnThisSeed && !(clusterRemovalInfos==edm::InputTag(""))) {
	    ClusterRemovalRefSetter refSetter(iEvent,clusterRemovalInfos);
	    TrajectorySeed::recHitContainer  newRecHitContainer;
	    newRecHitContainer.reserve(origSeedRef->nHits());
	    TrajectorySeed::const_iterator iH=origSeedRef->recHits().first;
	    TrajectorySeed::const_iterator iH_end=origSeedRef->recHits().second;
	    for (;iH!=iH_end;++iH){
	      newRecHitContainer.push_back(*iH);
	      refSetter.reKey(&newRecHitContainer.back());
	    }
	    outputSeeds->push_back( TrajectorySeed( origSeedRef->startingState(),
						    newRecHitContainer,
						    origSeedRef->direction()));
	  }
	  //doRekeyOnThisSeed=true
	  else{
	    //just copy the one we had before
	    outputSeeds->push_back( TrajectorySeed(*origSeedRef));
	  }
	  edm::Ref<TrajectorySeedCollection> pureRef(refTrajSeeds, outputSeeds->size()-1);
	  origSeedRef=edm::RefToBase<TrajectorySeed>( pureRef);
	}//creating a new seed and rekeying it rechit clusters.
	
	// Fill TrackExtra collection
	outputTrkExtras->push_back( reco::TrackExtra( 
						     track.outerPosition(), track.outerMomentum(), track.outerOk(),
						     track.innerPosition(), track.innerMomentum(), track.innerOk(),
						     track.outerStateCovariance(), track.outerDetId(),
						     track.innerStateCovariance(), track.innerDetId(),
						     track.seedDirection(), origSeedRef ) );
	seedsRefs[i]=origSeedRef;
	out_generalTracks->back().setExtra( reco::TrackExtraRef( refTrkExtras, outputTrkExtras->size() - 1) );
	reco::TrackExtra & tx = outputTrkExtras->back();
	tx.setResiduals(track.residuals());
	
	// fill TrackingRecHits
	unsigned nh1=track.recHitsSize();
	for ( unsigned ih=0; ih<nh1; ++ih ) { 
	  //const TrackingRecHit*hit=&((*(track->recHit(ih))));
	  outputTrkHits->push_back( track.recHit(ih)->clone() );
	  tx.add( TrackingRecHitRef( refTrkHits, outputTrkHits->size() - 1) );
	}
	
      }

      edm::Ref< std::vector<Trajectory> > trajRef(originalTrajHandle, i);
      TrajTrackAssociationCollection::const_iterator match = originalTrajTrackHandle->find(trajRef);
      if (match != originalTrajTrackHandle->end()) {
	if(curTrackRef.isNonnull()){
	  outputTrajs->push_back( *trajRef );
	  if (copyExtras_ && makeReKeyedSeeds_)
	    outputTrajs->back().setSeedRef( origSeedRef );
	  outputTTAss->insert(edm::Ref< std::vector<Trajectory> >(refTrajs, outputTrajs->size() - 1),curTrackRef );
	}
      }
    }
  }

  edm::ProductID nPID = refTrks.id();
  edm::TestHandle<TrackCollection> out_gtHandle(out_generalTracks.get(),nPID);

  fillerMVA.insert(out_gtHandle,mvaVec.begin(),mvaVec.end());
  fillerMVA.fill();

  iEvent.put(vmMVA,"MVAVals");
  iEvent.put(out_generalTracks);
  if (copyExtras_) {
    iEvent.put(outputTrkExtras);
    iEvent.put(outputTrkHits);
    if (makeReKeyedSeeds_)
      iEvent.put(outputSeeds);
  }
  iEvent.put(outputTrajs);
  iEvent.put(outputTTAss);
}

//------------------------------------------------------------
//------------------------------------------------------------
//------------------------------------------------------------
//------------------------------------------------------------
int DuplicateListMerger::matchCandidateToTrack(TrackCandidate candidate, edm::Handle<reco::TrackCollection > tracks){
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
