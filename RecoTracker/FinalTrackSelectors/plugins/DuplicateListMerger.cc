/** \class DuplicateListMerger
 * 
 * merges list of merge duplicate tracks with its parent list
 *
 * \author Matthew Walker
 */

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

#include "TMVA/Reader.h"

#include "trackAlgoPriorityOrder.h"

using namespace reco;

    class dso_hidden DuplicateListMerger final : public edm::stream::EDProducer<> {
       public:
         /// constructor
         explicit DuplicateListMerger(const edm::ParameterSet& iPara);
	 /// destructor
	 virtual ~DuplicateListMerger();

	 /// typedef container of candidate and input tracks
	 typedef std::pair<TrackCandidate,std::pair<reco::TrackRef,reco::TrackRef> > DuplicateRecord;
	 typedef edm::OwnVector<TrackingRecHit> RecHitContainer;
       protected:
	 /// produce one event
	 void produce( edm::Event &, const edm::EventSetup &) override;

       private:
	 int matchCandidateToTrack(TrackCandidate,edm::Handle<reco::TrackCollection>);

	 edm::ProductID clusterProductB( const TrackingRecHit *hit){
	   return reinterpret_cast<const BaseTrackerRecHit *>(hit)->firstClusterRef().id();
	 }

	 /// track input collection
         struct ThreeTokens {
            edm::InputTag tag;
            edm::EDGetTokenT<reco::TrackCollection> tk;
            edm::EDGetTokenT<std::vector<Trajectory> >        traj;
            edm::EDGetTokenT<TrajTrackAssociationCollection > tass;
            ThreeTokens() {}
            ThreeTokens(const edm::InputTag &tag_, edm::EDGetTokenT<reco::TrackCollection> && tk_, edm::EDGetTokenT<std::vector<Trajectory> > && traj_, edm::EDGetTokenT<TrajTrackAssociationCollection > && tass_) :
                tag(tag_), tk(tk_), traj(traj_), tass(tass_) {}
         };
         ThreeTokens threeTokens(const edm::InputTag &tag) {
            return ThreeTokens(tag, consumes<reco::TrackCollection>(tag), consumes<std::vector<Trajectory> >(tag), consumes<TrajTrackAssociationCollection >(tag));
         }
         ThreeTokens mergedTrackSource_, originalTrackSource_;
         edm::EDGetTokenT<edm::View<DuplicateRecord> > candidateSource_;

         edm::InputTag originalMVAVals_;
         edm::InputTag mergedMVAVals_;
         edm::EDGetTokenT<edm::ValueMap<float> > originalMVAValsToken_;
         edm::EDGetTokenT<edm::ValueMap<float> > mergedMVAValsToken_;

	 reco::TrackBase::TrackQuality qualityToSet_;
	 unsigned int diffHitsCut_;
	 float minTrkProbCut_;
	 bool copyExtras_;
	 bool makeReKeyedSeeds_;
     };
 

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/ClusterRemovalRefSetter.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "FWCore/Framework/interface/Event.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"


DuplicateListMerger::DuplicateListMerger(const edm::ParameterSet& iPara) 
{
  diffHitsCut_ = 9999;
  minTrkProbCut_ = 0.0;
  if(iPara.exists("diffHitsCut"))diffHitsCut_ = iPara.getParameter<int>("diffHitsCut");
  if(iPara.exists("minTrkProbCut"))minTrkProbCut_ = iPara.getParameter<double>("minTrkProbCut");
  if(iPara.exists("mergedSource")) mergedTrackSource_ = threeTokens(iPara.getParameter<edm::InputTag>("mergedSource"));
  if(iPara.exists("originalSource"))originalTrackSource_ = threeTokens(iPara.getParameter<edm::InputTag>("originalSource"));
  if(iPara.exists("candidateSource"))candidateSource_ = consumes<edm::View<DuplicateRecord> >(iPara.getParameter<edm::InputTag>("candidateSource"));


  if(iPara.exists("mergedMVAVals")){
    mergedMVAVals_ = iPara.getParameter<edm::InputTag>("mergedMVAVals");
  }else{
    mergedMVAVals_ = edm::InputTag(mergedTrackSource_.tag.label(),"MVAVals");
  }
  mergedMVAValsToken_ = consumes<edm::ValueMap<float> >(mergedMVAVals_);
  if(iPara.exists("originalMVAVals")){
    originalMVAVals_ = iPara.getParameter<edm::InputTag>("originalMVAVals");
  }else{
    originalMVAVals_ = edm::InputTag(originalTrackSource_.tag.label(),"MVAVals");
  }
  originalMVAValsToken_ = consumes<edm::ValueMap<float> >(originalMVAVals_);

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
  iEvent.getByToken(originalTrackSource_.tk,originalHandle);
  edm::Handle<reco::TrackCollection > mergedHandle;
  iEvent.getByToken(mergedTrackSource_.tk,mergedHandle);

  const reco::TrackCollection& mergedTracks(*mergedHandle);
  reco::TrackRefProd originalTrackRefs(originalHandle);
  reco::TrackRefProd mergedTrackRefs(mergedHandle);

  edm::Handle< std::vector<Trajectory> >  mergedTrajHandle;
  iEvent.getByToken(mergedTrackSource_.traj,mergedTrajHandle);
  edm::Handle< TrajTrackAssociationCollection >  mergedTrajTrackHandle;
  iEvent.getByToken(mergedTrackSource_.tass,mergedTrajTrackHandle);

  edm::Handle< std::vector<Trajectory> >  originalTrajHandle;
  iEvent.getByToken(originalTrackSource_.traj,originalTrajHandle);
  edm::Handle< TrajTrackAssociationCollection >  originalTrajTrackHandle;
  iEvent.getByToken(originalTrackSource_.tass,originalTrajTrackHandle);

  edm::Handle<edm::View<DuplicateRecord> > candidateHandle;
  iEvent.getByToken(candidateSource_,candidateHandle);

  std::auto_ptr<std::vector<reco::Track> > out_generalTracks(new std::vector<reco::Track>());
  out_generalTracks->reserve(originalHandle->size());
  reco::TrackRefProd refTrks = iEvent.getRefBeforePut<reco::TrackCollection>();
  std::auto_ptr< std::vector<Trajectory> > outputTrajs = std::auto_ptr< std::vector<Trajectory> >(new std::vector<Trajectory>());
  outputTrajs->reserve(originalTrajHandle->size()+mergedTrajHandle->size());
  edm::RefProd< std::vector<Trajectory> > refTrajs;
  //std::auto_ptr< TrajectorySeedCollection > outputSeeds

  std::auto_ptr<reco::TrackExtraCollection> outputTrkExtras;
  reco::TrackExtraRefProd refTrkExtras;
  std::auto_ptr<TrackingRecHitCollection> outputTrkHits;
  TrackingRecHitRefProd refTrkHits;
  std::auto_ptr<TrajectorySeedCollection> outputSeeds;
  edm::RefProd< TrajectorySeedCollection > refTrajSeeds;

  edm::Handle<edm::ValueMap<float> > originalMVAStore;
  edm::Handle<edm::ValueMap<float> > mergedMVAStore;

  iEvent.getByToken(originalMVAValsToken_,originalMVAStore);
  iEvent.getByToken(mergedMVAValsToken_,mergedMVAStore);

  std::auto_ptr<edm::ValueMap<float> > vmMVA(new edm::ValueMap<float>);
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

  std::auto_ptr< TrajTrackAssociationCollection >  outputTTAss = std::auto_ptr< TrajTrackAssociationCollection >(new TrajTrackAssociationCollection());

  for(matchIter0 = matches.begin(); matchIter0 != matches.end(); matchIter0++){
    reco::TrackRef inTrkRef1 = matchIter0->second->second.first;
    reco::TrackRef inTrkRef2 = matchIter0->second->second.second;
    const reco::Track& inTrk1 = *(inTrkRef1.get());
    const reco::Track& inTrk2 = *(inTrkRef2.get());
    reco::TrackBase::TrackAlgorithm newTrkAlgo = std::min(inTrk1.algo(),inTrk2.algo(),
                                                          [](reco::TrackBase::TrackAlgorithm a, reco::TrackBase::TrackAlgorithm b) {
                                                            return trackAlgoPriorityOrder[a] < trackAlgoPriorityOrder[b];
                                                          });
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
      out_generalTracks->back().setExtra( reco::TrackExtraRef( refTrkExtras, outputTrkExtras->size() - 1) );
      reco::TrackExtra & tx = outputTrkExtras->back();
      tx.setResiduals(track.residuals());
      // fill TrackingRecHits
      unsigned nh1=track.recHitsSize();
      auto const firstTrackIndex = outputTrkHits->size();
      for ( unsigned ih=0; ih<nh1; ++ih ) { 
	  //const TrackingRecHit*hit=&((*(track->recHit(ih))));
	outputTrkHits->push_back( track.recHit(ih)->clone() );
      }
      tx.setHits(  refTrkHits, firstTrackIndex, outputTrkHits->size() - firstTrackIndex );
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
	out_generalTracks->back().setExtra( reco::TrackExtraRef( refTrkExtras, outputTrkExtras->size() - 1) );
	reco::TrackExtra & tx = outputTrkExtras->back();
	tx.setResiduals(track.residuals());
	
	// fill TrackingRecHits
	unsigned nh1=track.recHitsSize();
       	tx.setHits(refTrkHits,outputTrkHits->size(),nh1);
	for (auto hh = track.recHitsBegin(), eh=track.recHitsEnd(); hh!=eh; ++hh ) { 
	  outputTrkHits->push_back( (*hh)->clone() );
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

  out_generalTracks->shrink_to_fit();  iEvent.put(out_generalTracks);
  if (copyExtras_) {
    outputTrkExtras->shrink_to_fit(); iEvent.put(outputTrkExtras);
    outputTrkHits->shrink_to_fit(); iEvent.put(outputTrkHits);
    if (makeReKeyedSeeds_) {
      outputSeeds->shrink_to_fit(); iEvent.put(outputSeeds);
    }
  }
  outputTrajs->shrink_to_fit(); iEvent.put(outputTrajs);
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
    if( (*tracks)[i].seedRef() != candidate.seedRef())continue;
    int match = 0;
    trackingRecHit_iterator trackRecBegin = (*tracks)[i].recHitsBegin();
    trackingRecHit_iterator trackRecEnd = (*tracks)[i].recHitsEnd();
    for(;trackRecBegin != trackRecEnd; trackRecBegin++){
      if(std::find(rawIds.begin(),rawIds.end(),(*(trackRecBegin))->rawId()) != rawIds.end()) match++;
    }
    if(match != (int)( (*tracks)[i].recHitsSize() ) ) continue;
    track = i;
  }

  return track;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(DuplicateListMerger);
