//
// Package:         RecoTracker/FinalTrackSelectors
// Class:           TrackListMerger
// 
// Description:     TrackList Cleaner and Merger able to deal with many lists
//
// Original Author: David Lange
// Created:         April 4, 2011
//


#include <memory>
#include <string>
#include <iostream>
#include <cmath>
#include <vector>

#include "RecoTracker/FinalTrackSelectors/src/TrackListMerger.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1DCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

//#include "DataFormats/TrackReco/src/classes.h"

#include "RecoTracker/TrackProducer/interface/ClusterRemovalRefSetter.h"


namespace {
#ifdef STAT_TSB
  struct StatCount {
    long long totBegin=0;
    long long totPre=0;
    long long totEnd=0;
    void begin(int tt) {
      totBegin+=tt;
    }
    void pre(int tt) { totPre+=tt;}
    void end(int tt) { totEnd+=tt;}


    void print() const {
      std::cout << "TrackListMerger stat\nBegin/Pre/End/ "
    		<<  totBegin <<'/'<< totPre <<'/'<< totEnd
		<< std::endl;
    }
    StatCount() {}
    ~StatCount() { print();}
  };

#else
  struct StatCount {
    void begin(int){}
    void pre(int){}
    void end(int){}
  };
#endif

  StatCount statCount;

}



namespace cms
{
  
  edm::ProductID clusterProductB( const TrackingRecHit *hit){
    return reinterpret_cast<const BaseTrackerRecHit *>(hit)->firstClusterRef().id();
  }
  


  TrackListMerger::TrackListMerger(edm::ParameterSet const& conf) {
    copyExtras_ = conf.getUntrackedParameter<bool>("copyExtras", true);
    
    trackProducers_ = conf.getParameter<std::vector<edm::InputTag> >("TrackProducers");
    //which of these do I need to turn into vectors?
    maxNormalizedChisq_ =  conf.getParameter<double>("MaxNormalizedChisq");
    minPT_ =  conf.getParameter<double>("MinPT");
    minFound_ = (unsigned int)conf.getParameter<int>("MinFound");
    epsilon_ =  conf.getParameter<double>("Epsilon");
    shareFrac_ =  conf.getParameter<double>("ShareFrac");
    allowFirstHitShare_ = conf.getParameter<bool>("allowFirstHitShare");
    foundHitBonus_ = conf.getParameter<double>("FoundHitBonus");
    lostHitPenalty_ = conf.getParameter<double>("LostHitPenalty");
    std::string qualityStr = conf.getParameter<std::string>("newQuality");
    
    if (qualityStr != "") {
      qualityToSet_ = reco::TrackBase::qualityByName(conf.getParameter<std::string>("newQuality"));
    }
    else 
      qualityToSet_ = reco::TrackBase::undefQuality;
    
    use_sharesInput_ = true;
    if ( epsilon_ > 0.0 )use_sharesInput_ = false;
    
    edm::VParameterSet setsToMerge=conf.getParameter<edm::VParameterSet>("setsToMerge");
    
    for ( unsigned int i=0; i<setsToMerge.size(); i++) { 
      listsToMerge_.push_back(setsToMerge[i].getParameter<std::vector< int> >("tLists"));   
      promoteQuality_.push_back(setsToMerge[i].getParameter<bool>("pQual"));   
    }
    
    hasSelector_=conf.getParameter<std::vector<int> >("hasSelector");
    selectors_=conf.getParameter<std::vector<edm::InputTag> >("selectedTrackQuals");   
    
    trkQualMod_=conf.getParameter<bool>("writeOnlyTrkQuals");
    if ( trkQualMod_) {
      bool ok=true;
      for ( unsigned int i=1; i<trackProducers_.size(); i++) {
	if (!(trackProducers_[i]==trackProducers_[0])) ok=false;
      }
      if ( !ok) {
	throw cms::Exception("Bad input") << "to use writeOnlyTrkQuals=True all input InputTags must be the same";
      }
      produces<edm::ValueMap<int> >();
    }
    else{
      produces<reco::TrackCollection>();
      
      makeReKeyedSeeds_ = conf.getUntrackedParameter<bool>("makeReKeyedSeeds",false);
      if (makeReKeyedSeeds_){
	copyExtras_=true;
	produces<TrajectorySeedCollection>();
      }
      
      if (copyExtras_) {
        produces<reco::TrackExtraCollection>();
        produces<TrackingRecHitCollection>();
      }
      produces< std::vector<Trajectory> >();
      produces< TrajTrackAssociationCollection >();
    }
    
  }
  
  
  // Virtual destructor needed.
  TrackListMerger::~TrackListMerger() { }  
  
  // Functions that gets called by framework every event
  void TrackListMerger::produce(edm::Event& e, const edm::EventSetup& es)
  {
    // extract tracker geometry
    //
    //edm::ESHandle<TrackerGeometry> theG;
    //es.get<TrackerDigiGeometryRecord>().get(theG);
    
    //    using namespace reco;
    
    // get Inputs 
    // if 1 input list doesn't exist, make an empty list, issue a warning, and continue
    // this allows TrackListMerger to be used as a cleaner only if handed just one list
    // if both input lists don't exist, will issue 2 warnings and generate an empty output collection
    // 
    static const reco::TrackCollection s_empty;
    
    std::vector<const reco::TrackCollection *> trackColls;
    std::vector<edm::Handle<reco::TrackCollection> > trackHandles(trackProducers_.size());
    for ( unsigned int i=0; i<trackProducers_.size(); i++) {
      trackColls.push_back(0);
      //edm::Handle<reco::TrackCollection> trackColl;
      e.getByLabel(trackProducers_[i], trackHandles[i]);
      if (trackHandles[i].isValid()) {
	trackColls[i]= trackHandles[i].product();
      } else {
	edm::LogWarning("TrackListMerger") << "TrackCollection " << trackProducers_[i] <<" not found";
	trackColls[i]=&s_empty;
      }
    }
    
    unsigned int collsSize =trackColls.size();
    unsigned int rSize=0;
    unsigned int trackCollSizes[collsSize];
    unsigned int trackCollFirsts[collsSize];
    for (unsigned int i=0; i!=collsSize; i++) {
      trackCollSizes[i]=trackColls[i]->size();
      trackCollFirsts[i]=rSize;
      rSize+=trackCollSizes[i];
    }
    
    statCount.begin(rSize);

    //
    //  quality cuts first
    // 
    int i=-1;
    
    int selected[rSize];
    int indexG[rSize];
    bool trkUpdated[rSize]; 
    int trackCollNum[rSize];
    int trackQuals[rSize];
    for (unsigned int j=0; j<rSize;j++) {
      indexG[j]=-1; selected[j]=1; trkUpdated[j]=false; trackCollNum[j]=0; trackQuals[j]=0;
    }

    int ngood=0;
    for (unsigned int j=0; j!= collsSize; j++) {
      const reco::TrackCollection *tC1=trackColls[j];
      
      edm::Handle<edm::ValueMap<int> > trackSelColl;
      if ( hasSelector_[j]>0 ){
	e.getByLabel(selectors_[j], trackSelColl);
      }
      
      if ( 0<tC1->size() ){
	unsigned int iC=0;
	for (reco::TrackCollection::const_iterator track=tC1->begin(); track!=tC1->end(); track++){
	  i++; 
	  trackCollNum[i]=j;
	  trackQuals[i]=track->qualityMask();
	  
	  if ( hasSelector_[j]>0 ) {
	    reco::TrackRef trkRef=reco::TrackRef(trackHandles[j],iC);
	    int qual=(*trackSelColl)[trkRef];
	    if ( qual < 0 ) {
	      selected[i]=0;
	      iC++;
	      continue;
	    }
	    else{
	      trackQuals[i]=qual;
	    }
	  }
	  iC++;
	  selected[i]=trackQuals[i]+10;//10 is magic number used throughout...
	  if ((short unsigned)track->ndof() < 1){
	    selected[i]=0; 
	    continue;
	  }
	  if (track->normalizedChi2() > maxNormalizedChisq_){
	    selected[i]=0; 
	    continue;
	  }
	  if (track->found() < minFound_){
	    selected[i]=0; 
	    continue;
	  }
	  if (track->pt() < minPT_){
	    selected[i]=0; 
	    continue;
	  }
	  // good!
	  indexG[i] = ngood++;
	  //if ( beVerb) std::cout << "inverb " << track->pt() << " " << selected[i] << std::endl;
	}//end loop over tracks
      }//end more than 0 track
    } // loop over trackcolls
    
    
    statCount.pre(ngood);    

    //cache the rechits and valid hits
    std::vector<const TrackingRecHit*> rh1[ngood];  // an array of vectors!
    reco::PatternSet<23> pattern[ngood];
    unsigned char algo[ngood];
    // short int validHits[ngood];
    // short int lostHits[ngood];
    float score[ngood];
    for ( unsigned int j=0; j<rSize; j++) {
      if (selected[j]==0) continue;
      int i = indexG[j];
      assert(i>=0);
      unsigned int collNum=trackCollNum[j];
      unsigned int trackNum=j-trackCollFirsts[collNum];
      const reco::Track *track=&((trackColls[collNum])->at(trackNum)); 

      algo[i]=track->algo();
      int validHits=track->numberOfValidHits();
      int lostHits=track->numberOfLostHits();
      score[i] = foundHitBonus_*validHits - lostHitPenalty_*lostHits - track->chi2();
      pattern[i].fill(track->hitPattern());

      rh1[i].reserve(validHits) ; // track->recHitsSize());
      for (trackingRecHit_iterator it = track->recHitsBegin();  it != track->recHitsEnd(); ++it) { 
	const TrackingRecHit* hit = &(**it);
	if likely(hit->isValid()) rh1[i].push_back(hit);
      }
    }
    
    //DL here
    if likely(ngood>2 && collsSize>1)
    for ( unsigned int ltm=0; ltm<listsToMerge_.size(); ltm++) {
      int saveSelected[rSize];
      bool notActive[collsSize];
      for (unsigned int cn=0;cn!=collsSize;++cn)
	notActive[cn]= find(listsToMerge_[ltm].begin(),listsToMerge_[ltm].end(),cn)==listsToMerge_[ltm].end();

      for ( unsigned int i=0; i<rSize; i++) saveSelected[i]=selected[i];
      
      //DL protect against 0 tracks? 
      for ( unsigned int i=0; i<trackCollFirsts[collsSize-1]; i++) {
	if (selected[i]==0) continue;
	unsigned int collNum=trackCollNum[i];
	//nothing to do if this is the last collection
	assert(collNum!=collsSize-1 );
	
	//check that this track is in one of the lists for this iteration
	if (notActive[collNum]) continue;

	int k1 = indexG[i];
	unsigned nh1=rh1[k1].size();
	int qualityMaskT1 = trackQuals[i];
	
	int nhit1 = nh1; // validHits[k1];
	float score1 = score[k1];
	
	// start at next collection
	for ( unsigned int j=trackCollFirsts[collNum+1]; j<rSize; j++) {
	  if (selected[j]==0) continue;
	  unsigned int collNum2=trackCollNum[j];
	  assert ( collNum != collNum2);
	  
	  //check that this track is in one of the lists for this iteration
	  if (notActive[collNum2]) continue;

	  int k2 = indexG[j];
	  	  
	  int newQualityMask = -9; //avoid resetting quality mask if not desired 10+ -9 =1
	  if (promoteQuality_[ltm]) {
	    int maskT1= saveSelected[i]>1? saveSelected[i]-10 : qualityMaskT1;
	    int maskT2= saveSelected[j]>1? saveSelected[j]-10 : trackQuals[j];
	    newQualityMask =(maskT1 | maskT2); // take OR of trackQuality 
	  }
	  unsigned int nh2=rh1[k2].size();
	  int nhit2 = nh2; // validHits[k2];

	  // do not even bother if not enough "pattern in common"
	  int ncomm = reco::commonHits(pattern[k1],pattern[k2]).size();
	  if (ncomm<(std::min(nhit1,nhit2)-1)*shareFrac_) continue;

	  //loop over rechits
	  int noverlap=0;
	  int firstoverlap=0;
	  
	  unsigned int js=0;
	  for ( unsigned int ih=0; ih<nh1; ++ih ) { 
	    const TrackingRecHit* it = rh1[k1][ih];
	    // if unlikely(!it->isValid()) continue;
	    if (js==nh2) break;
	    for ( unsigned int jh=js; jh<nh2; ++jh ) { 
	      const TrackingRecHit *jt=rh1[k2][jh];
	      // if unlikely(!jt->isValid() ) continue;
	      if ( (it->geographicalId()|3) !=(jt->geographicalId()|3) ) continue;  // VI: mask mono/stereo...
	      if (!use_sharesInput_){
		float delta = std::abs ( it->localPosition().x()-jt->localPosition().x() ); 
		if ((it->geographicalId()==jt->geographicalId())&&(delta<epsilon_)) {
		  noverlap++;
		  if ( allowFirstHitShare_ && ( ih == 0 ) && ( jh == 0 ) ) firstoverlap=1;
		  js=jh+1;
		  break;
		}
	      }else{
		if ( it->sharesInput(jt,TrackingRecHit::some) ) {
		  noverlap++;
		  if ( allowFirstHitShare_ && ( ih == 0 ) && ( jh == 0 ) ) firstoverlap=1;
		  js=jh+1;
		  break;
		} // tracks share input
	      } //else use_sharesInput
	    } // rechits on second track  
	  } //loop over ih (rechits on first track
	  
	  
	  if ( (noverlap-firstoverlap) > (std::min(nhit1,nhit2)-firstoverlap)*shareFrac_ ) {
	    float score2 = score[k2];
	    constexpr float almostSame = 1.001f;
	    if ( score1 > almostSame * score2 ) {
	      selected[j]=0;
	      selected[i]=10+newQualityMask; // add 10 to avoid the case where mask = 1
	      trkUpdated[i]=true;
	    } else if ( score2 > almostSame * score1 ) {
	      selected[i]=0;
	      selected[j]=10+newQualityMask;  // add 10 to avoid the case where mask = 1
	      trkUpdated[j]=true;
	    }else{
	      // If tracks from both iterations are virtually identical, choose the one from the first iteration.
	      if (algo[k1] <= algo[k2]) {
		selected[j]=0;
		selected[i]=10+newQualityMask; // add 10 to avoid the case where mask = 1
		trkUpdated[i]=true;
	      }else{
		selected[i]=0;
		selected[j]=10+newQualityMask; // add 10 to avoid the case where mask = 1
		trkUpdated[j]=true;
	      }
	    }//end fi < fj
	  }//end got a duplicate
	  //stop if the ith track is now unselected
	  if (selected[i]==0) break;
	}//end track2 loop
      }//end track loop
    } //end loop over track list sets
    
   

 
    
    // special case - if just doing the trkquals 
    if (trkQualMod_) {
      std::auto_ptr<edm::ValueMap<int> > vm = std::auto_ptr<edm::ValueMap<int> >(new edm::ValueMap<int>);
      edm::ValueMap<int>::Filler filler(*vm);
      
      unsigned int tSize=trackColls[0]->size();
      std::vector<int> finalQuals(tSize,-1); //default is unselected
      for ( unsigned int i=0; i<rSize; i++) {
	unsigned int tNum=i%tSize;
	
	if (selected[i]>1 ) { 
	  finalQuals[tNum]=selected[i]-10;
	  if (trkUpdated[i])
	    finalQuals[tNum]=(finalQuals[tNum] | (1<<qualityToSet_));
	}
	if ( selected[i]==1 )
	  finalQuals[tNum]=trackQuals[i];
      }
      
      filler.insert(trackHandles[0], finalQuals.begin(),finalQuals.end());
      filler.fill();
      
      e.put(vm);
      return;
    }
    
    
    //
    //  output selected tracks - if any
    //
    
    trackRefs.resize(rSize);
    std::vector<edm::RefToBase<TrajectorySeed> > seedsRefs(rSize);
    
    unsigned int nToWrite=0;
    for ( unsigned int i=0; i<rSize; i++) 
      if (selected[i]!=0) nToWrite++;
    
    
    outputTrks = std::auto_ptr<reco::TrackCollection>(new reco::TrackCollection);
    outputTrks->reserve(nToWrite);
    refTrks = e.getRefBeforePut<reco::TrackCollection>();      
    
    if (copyExtras_) {
      outputTrkExtras = std::auto_ptr<reco::TrackExtraCollection>(new reco::TrackExtraCollection);
      outputTrkExtras->reserve(nToWrite);
      refTrkExtras    = e.getRefBeforePut<reco::TrackExtraCollection>();
      outputTrkHits   = std::auto_ptr<TrackingRecHitCollection>(new TrackingRecHitCollection);
      outputTrkHits->reserve(nToWrite*25);
      refTrkHits      = e.getRefBeforePut<TrackingRecHitCollection>();
      if (makeReKeyedSeeds_){
	outputSeeds = std::auto_ptr<TrajectorySeedCollection>(new TrajectorySeedCollection);
	outputSeeds->reserve(nToWrite);
	refTrajSeeds = e.getRefBeforePut<TrajectorySeedCollection>();
      }
    }
    

    outputTrajs = std::auto_ptr< std::vector<Trajectory> >(new std::vector<Trajectory>()); 
    outputTrajs->reserve(rSize);
    outputTTAss = std::auto_ptr< TrajTrackAssociationCollection >(new TrajTrackAssociationCollection());
    
    
    
    for ( unsigned int i=0; i<rSize; i++) {
      if (selected[i]==0) {
	trackRefs[i]=reco::TrackRef();
	continue;
      }
      
      unsigned int collNum=trackCollNum[i];
      unsigned int trackNum=i-trackCollFirsts[collNum];
      const reco::Track *track=&((trackColls[collNum])->at(trackNum)); 
      outputTrks->push_back( reco::Track( *track ) );
      if (selected[i]>1 ) { 
	outputTrks->back().setQualityMask(selected[i]-10);
	if (trkUpdated[i])
	  outputTrks->back().setQuality(qualityToSet_);
      }
      //might duplicate things, but doesnt hurt
      if ( selected[i]==1 )
	outputTrks->back().setQualityMask(trackQuals[i]);
      
      // if ( beVerb ) std::cout << "selected " << outputTrks->back().pt() << " " << outputTrks->back().qualityMask() << " " << selected[i] << std::endl;
      
      //fill the TrackCollection
      if (copyExtras_) {
	edm::RefToBase<TrajectorySeed> origSeedRef = track->seedRef();
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
	      edm::Provenance prov=e.getProvenance(pID);
	      clusterRemovalInfos=edm::InputTag(prov.moduleLabel(),
						prov.productInstanceName(),
						prov.processName());
	      doRekeyOnThisSeed=e.getByLabel(clusterRemovalInfos,CRIh);
	    }//valid hit
	  }//nhit!=0
	  
	  if (doRekeyOnThisSeed && !(clusterRemovalInfos==edm::InputTag(""))) {
	    ClusterRemovalRefSetter refSetter(e,clusterRemovalInfos);
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
						     track->outerPosition(), track->outerMomentum(), track->outerOk(),
						     track->innerPosition(), track->innerMomentum(), track->innerOk(),
						     track->outerStateCovariance(), track->outerDetId(),
						     track->innerStateCovariance(), track->innerDetId(),
						     track->seedDirection(), origSeedRef ) );
	seedsRefs[i]=origSeedRef;
	outputTrks->back().setExtra( reco::TrackExtraRef( refTrkExtras, outputTrkExtras->size() - 1) );
	reco::TrackExtra & tx = outputTrkExtras->back();
	tx.setResiduals(track->residuals());
	
	// fill TrackingRecHits
	unsigned nh1=track->recHitsSize();
	for ( unsigned ih=0; ih<nh1; ++ih ) { 
	  //const TrackingRecHit*hit=&((*(track->recHit(ih))));
	  outputTrkHits->push_back( track->recHit(ih)->clone() );
	  tx.add( TrackingRecHitRef( refTrkHits, outputTrkHits->size() - 1) );
	}
      }
      trackRefs[i] = reco::TrackRef(refTrks, outputTrks->size() - 1);
      
      
    }//end faux loop over tracks
    
    //Fill the trajectories, etc. for 1st collection
    refTrajs    = e.getRefBeforePut< std::vector<Trajectory> >();
  
    for (unsigned int ti=0; ti<trackColls.size(); ti++) {
      edm::Handle< std::vector<Trajectory> >  hTraj1;
      edm::Handle< TrajTrackAssociationCollection >  hTTAss1;
      e.getByLabel(trackProducers_[ti], hTraj1);
      e.getByLabel(trackProducers_[ti], hTTAss1);
      
      if (hTraj1.failedToGet() || hTTAss1.failedToGet()) continue;
      
      for (size_t i = 0, n = hTraj1->size(); i < n; ++i) {
      edm::Ref< std::vector<Trajectory> > trajRef(hTraj1, i);
      TrajTrackAssociationCollection::const_iterator match = hTTAss1->find(trajRef);
      if (match != hTTAss1->end()) {
	const edm::Ref<reco::TrackCollection> &trkRef = match->val; 
	short oldKey = trackCollFirsts[ti]+static_cast<short>(trkRef.key());
	if (trackRefs[oldKey].isNonnull()) {
	  outputTrajs->push_back( *trajRef );
	  //if making extras and the seeds at the same time, change the seed ref on the trajectory
	  if (copyExtras_ && makeReKeyedSeeds_)
	    outputTrajs->back().setSeedRef( seedsRefs[oldKey] );
	  outputTTAss->insert ( edm::Ref< std::vector<Trajectory> >(refTrajs, outputTrajs->size() - 1), 
				trackRefs[oldKey] );
	}
      }
      }
    }
    
    statCount.end(outputTrks->size());


    e.put(outputTrks);
    if (copyExtras_) {
      e.put(outputTrkExtras);
      e.put(outputTrkHits);
      if (makeReKeyedSeeds_)
	e.put(outputSeeds);
    }
    e.put(outputTrajs);
    e.put(outputTTAss);
    return;
  
  }//end produce
}
