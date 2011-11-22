//
// Package:         RecoTracker/FinalTrackSelectors
// Class:           TrackListMerger
// 
// Description:     TrackList Cleaner and Merger able to deal with many lists
//
// Original Author: David Lange
// Created:         April 4, 2011
//
// $Author: dlange $
// $Date: 2011/05/28 02:33:06 $
// $Revision: 1.5 $
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


namespace cms
{
  
  edm::ProductID clusterProductB( const TrackingRecHit *hit){
    edm::ProductID pID;
    //cast it into the proper class	and find productID
    DetId detid = hit->geographicalId(); 
    uint32_t subdet = detid.subdetId();
    if ((subdet == PixelSubdetector::PixelBarrel) || (subdet == PixelSubdetector::PixelEndcap)) {
      pID=reinterpret_cast<const SiPixelRecHit *>(hit)->cluster().id();
    } else {
      const std::type_info &type = typeid(*hit);
      if (type == typeid(SiStripRecHit2D)) {
	pID=reinterpret_cast<const SiStripRecHit2D *>(hit)->cluster().id();
      } else if (type == typeid(SiStripRecHit1D)) {
	pID=reinterpret_cast<const SiStripRecHit1D *>(hit)->cluster().id();	
      } else if (type == typeid(SiStripMatchedRecHit2D)) {
	const SiStripMatchedRecHit2D *mhit = reinterpret_cast<const SiStripMatchedRecHit2D *>(hit);
	pID=mhit->monoHit()->cluster().id();
      } else if (type == typeid(ProjectedSiStripRecHit2D)) {
	const ProjectedSiStripRecHit2D *phit = reinterpret_cast<const ProjectedSiStripRecHit2D *>(hit);
	pID=(&phit->originalHit())->cluster().id();
      } else throw cms::Exception("Unknown RecHit Type") << "RecHit of type " << type.name() << " not supported. (use c++filt to demangle the name)";
    }
        
    return pID;}
  


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

    unsigned int rSize=0;
    std::vector<unsigned int> trackCollSizes(trackColls.size(),0);
    std::vector<unsigned int> trackCollFirsts(trackColls.size(),0);
    for (unsigned int i=0; i<trackColls.size(); i++) {
      trackCollSizes[i]=trackColls[i]->size();
      trackCollFirsts[i]=rSize;
      rSize+=trackCollSizes[i];
    }


  //
  //  quality cuts first
  // 
    int i=-1;

    std::vector<int> selected(rSize,1); 
    std::vector<bool> trkUpdated(rSize,false); 
    std::vector<int> trackCollNum(rSize,0);
    std::vector<int> trackQuals(rSize,0);

    for (unsigned int j=0; j< trackColls.size(); j++) {
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
	  //if ( beVerb) std::cout << "inverb " << track->pt() << " " << selected[i] << std::endl;
	}//end loop over tracks
      }//end more than 0 track
    } // loop over trackcolls

   

    //cache the rechits and valid hits
    std::vector<std::vector<const TrackingRecHit*> > rh1(rSize);
    std::vector<int> validHits(rSize,0);
    for ( unsigned int i=0; i<rSize; i++) {
      if (selected[i]==0) continue;
      unsigned int collNum=trackCollNum[i];
      unsigned int trackNum=i-trackCollFirsts[collNum];
      const reco::Track *track=&((trackColls[collNum])->at(trackNum)); 
      validHits[i]=track->numberOfValidHits();

      rh1[i].reserve(track->recHitsSize());
      for (trackingRecHit_iterator it = track->recHitsBegin();  it != track->recHitsEnd(); ++it) { 
	const TrackingRecHit* hit = &(**it);
	rh1[i].push_back(hit);
      }
    }

    //DL here    
    for ( unsigned int ltm=0; ltm<listsToMerge_.size(); ltm++) {
      if ( rSize==0 ) continue;
      std::vector<int> saveSelected(rSize);
      for ( unsigned int i=0; i<rSize; i++) saveSelected[i]=selected[i];

      //DL protect against 0 tracks? 
      for ( unsigned int i=0; i<rSize-1; i++) {
	if (selected[i]==0) continue;
	unsigned int collNum=trackCollNum[i];
	//nothing to do if this is the last collection
	if (collNum==trackCollNum.size()-1) continue;

	//check that this track is in one of the lists for this iteration
	std::vector<int>::iterator isActive=find(listsToMerge_[ltm].begin(),listsToMerge_[ltm].end(),collNum);
	if ( isActive==listsToMerge_[ltm].end() ) continue;
	unsigned int trackNum=i-trackCollFirsts[collNum];
	const reco::Track *track=&((trackColls[collNum])->at(trackNum)); 
	unsigned nh1=rh1[i].size();
	int qualityMaskT1 = trackQuals[i];

	int nhit1 = validHits[i];

	for ( unsigned int j=i+1; j<rSize; j++) {
	  if (selected[j]==0) continue;
	  unsigned int collNum2=trackCollNum[j];
	  if ( collNum == collNum2) continue;

	  //check that this track is in one of the lists for this iteration
	  std::vector<int>::iterator isActive=find(listsToMerge_[ltm].begin(),listsToMerge_[ltm].end(),collNum2);
	  if ( isActive==listsToMerge_[ltm].end() ) continue;
	  
	  unsigned int trackNum2=j-trackCollFirsts[collNum2];
	  const reco::Track *track2=&((trackColls[collNum2])->at(trackNum2)); 
	  
	  //loop over rechits
	  int noverlap=0;
	  int firstoverlap=0;
	  unsigned nh2=track2->recHitsSize();
	  
	  for ( unsigned ih=0; ih<nh1; ++ih ) { 
	    const TrackingRecHit* it = rh1[i][ih];
	    if (!it->isValid()) continue;
	    for ( unsigned jh=0; jh<nh2; ++jh ) { 
	      const TrackingRecHit *jt=rh1[j][jh];
	      if (!jt->isValid() ) continue;
	      
	      if (!use_sharesInput_){
		float delta = fabs ( it->localPosition().x()-jt->localPosition().x() ); 
		if ((it->geographicalId()==jt->geographicalId())&&(delta<epsilon_)) {
		  noverlap++;
		  if ( allowFirstHitShare_ && ( ih == 0 ) && ( jh == 0 ) ) firstoverlap=1;
		}
	      }else{
		if ( it->sharesInput(jt,TrackingRecHit::some) ) {
		  noverlap++;
		  if ( allowFirstHitShare_ && ( ih == 0 ) && ( jh == 0 ) ) firstoverlap=1;
		} // tracks share input
	      } //else use_sharesInput
	    } // rechits on second track  
	  } //loop over ih (rechits on first track
	  
	  int newQualityMask = -9; //avoid resetting quality mask if not desired 10+ -9 =1
	  if (promoteQuality_[ltm]) {
	    int maskT1= saveSelected[i]>1? saveSelected[i]-10 : qualityMaskT1;
	    int maskT2= saveSelected[j]>1? saveSelected[j]-10 : trackQuals[j];
	    newQualityMask =(maskT1 | maskT2); // take OR of trackQuality 
	  }
	  int nhit2 = validHits[j];

	  if ( (noverlap-firstoverlap) > (std::min(nhit1,nhit2)-firstoverlap)*shareFrac_ ) {
	    if ( nhit1 > nhit2 ){
	      selected[j]=0; 
	      selected[i]=10+newQualityMask; // add 10 to avoid the case where mask = 1
	      trkUpdated[i]=true;
	    }else{
	      if ( nhit1 < nhit2 ){
		selected[i]=0; 
		selected[j]=10+newQualityMask;  // add 10 to avoid the case where mask = 1
		trkUpdated[j]=true;
	      }else{
		const double almostSame = 1.001;
		if (track->normalizedChi2() > almostSame * track2->normalizedChi2()) {
		  selected[i]=0;
		  selected[j]=10+newQualityMask; // add 10 to avoid the case where mask = 1
		  trkUpdated[j]=true;
		}else if (track2->normalizedChi2() > almostSame * track->normalizedChi2()) {
		  selected[j]=0;
		  selected[i]=10+newQualityMask; // add 10 to avoid the case where mask = 1
		  trkUpdated[i]=true;
		}else{
		  // If tracks from both iterations are virtually identical, choose the one from the first iteration.
		  if (track->algo() <= track2->algo()) {
		    selected[j]=0;
		    selected[i]=10+newQualityMask; // add 10 to avoid the case where mask = 1
		    trkUpdated[i]=true;
		  }else{
		    selected[i]=0;
		    selected[j]=10+newQualityMask; // add 10 to avoid the case where mask = 1
		    trkUpdated[j]=true;
		  }
		}
	      }//end fi > or = fj
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
