//
// Package:         RecoTracker/FinalTrackSelectors
// Class:           SimpleTrackListMerger
//
// Description:     TrackList Cleaner and Merger
//
// Original Author: Steve Wagner, stevew@pizero.colorado.edu
// Created:         Sat Jan 14 22:00:00 UTC 2006
//
//

#include <memory>
#include <string>
#include <iostream>
#include <cmath>
#include <vector>

#include "RecoTracker/FinalTrackSelectors/interface/SimpleTrackListMerger.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1DCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

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
  // VI January 2012   to be migrated to omnicluster (or firstCluster)
  edm::ProductID clusterProduct( const TrackingRecHit *hit){
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
	pID=mhit->monoClusterRef().id();
      } else if (type == typeid(ProjectedSiStripRecHit2D)) {
	const ProjectedSiStripRecHit2D *phit = reinterpret_cast<const ProjectedSiStripRecHit2D *>(hit);
	pID= phit->cluster().id();
      } else throw cms::Exception("Unknown RecHit Type") << "RecHit of type " << type.name() << " not supported. (use c++filt to demangle the name)";
    }

    return pID;}



  SimpleTrackListMerger::SimpleTrackListMerger(edm::ParameterSet const& conf) :
    conf_(conf)
  {
    copyExtras_ = conf_.getUntrackedParameter<bool>("copyExtras", true);

    produces<reco::TrackCollection>();

    makeReKeyedSeeds_ = conf_.getUntrackedParameter<bool>("makeReKeyedSeeds",false);
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

    trackProducer1 = conf_.getParameter<std::string>("TrackProducer1");
    trackProducer2 = conf_.getParameter<std::string>("TrackProducer2");

    trackProducer1Token = consumes<reco::TrackCollection>(trackProducer1);
    trackProducer2Token = consumes<reco::TrackCollection>(trackProducer2);
    trackProducer1TrajToken = consumes< std::vector<Trajectory> >(trackProducer1);
    trackProducer2TrajToken = consumes< std::vector<Trajectory> >(trackProducer2);
    trackProducer1AssToken = consumes< TrajTrackAssociationCollection >(trackProducer1);
    trackProducer2AssToken = consumes< TrajTrackAssociationCollection >(trackProducer2);

  }


  // Virtual destructor needed.
  SimpleTrackListMerger::~SimpleTrackListMerger() { }

  // Functions that gets called by framework every event
  void SimpleTrackListMerger::produce(edm::Event& e, const edm::EventSetup& es)
  {
    // retrieve producer name of input TrackCollection(s)

    double maxNormalizedChisq =  conf_.getParameter<double>("MaxNormalizedChisq");
    double minPT =  conf_.getParameter<double>("MinPT");
    unsigned int minFound = (unsigned int)conf_.getParameter<int>("MinFound");
    double epsilon =  conf_.getParameter<double>("Epsilon");
    bool use_sharesInput = true;
    if ( epsilon > 0.0 )use_sharesInput = false;
    double shareFrac =  conf_.getParameter<double>("ShareFrac");
    double foundHitBonus = conf_.getParameter<double>("FoundHitBonus");
    double lostHitPenalty = conf_.getParameter<double>("LostHitPenalty");

    bool promoteQuality = conf_.getParameter<bool>("promoteTrackQuality");
    bool allowFirstHitShare = conf_.getParameter<bool>("allowFirstHitShare");
//

    // New track quality should be read from the file
    std::string qualityStr = conf_.getParameter<std::string>("newQuality");
    reco::TrackBase::TrackQuality qualityToSet;
    if (qualityStr != "") {
      qualityToSet = reco::TrackBase::qualityByName(conf_.getParameter<std::string>("newQuality"));
    }
    else
      qualityToSet = reco::TrackBase::undefQuality;

    // extract tracker geometry
    //
    edm::ESHandle<TrackerGeometry> theG;
    es.get<TrackerDigiGeometryRecord>().get(theG);

//    using namespace reco;

    // get Inputs
    // if 1 input list doesn't exist, make an empty list, issue a warning, and continue
    // this allows SimpleTrackListMerger to be used as a cleaner only if handed just one list
    // if both input lists don't exist, will issue 2 warnings and generate an empty output collection
    //
    const reco::TrackCollection *TC1 = 0;
    static const reco::TrackCollection s_empty1, s_empty2;
    edm::Handle<reco::TrackCollection> trackCollection1;
    e.getByToken(trackProducer1Token, trackCollection1);
    if (trackCollection1.isValid()) {
      TC1 = trackCollection1.product();
      //std::cout << "1st collection " << trackProducer1 << " has "<< TC1->size() << " tracks" << std::endl ;
    } else {
      TC1 = &s_empty1;
      edm::LogWarning("SimpleTrackListMerger") << "1st TrackCollection " << trackProducer1 << " not found; will only clean 2nd TrackCollection " << trackProducer2 ;
    }
    const reco::TrackCollection tC1 = *TC1;

    const reco::TrackCollection *TC2 = 0;
    edm::Handle<reco::TrackCollection> trackCollection2;
    e.getByToken(trackProducer2Token, trackCollection2);
    if (trackCollection2.isValid()) {
      TC2 = trackCollection2.product();
      //std::cout << "2nd collection " << trackProducer2 << " has "<< TC2->size() << " tracks" << std::endl ;
    } else {
        TC2 = &s_empty2;
        edm::LogWarning("SimpleTrackListMerger") << "2nd TrackCollection " << trackProducer2 << " not found; will only clean 1st TrackCollection " << trackProducer1 ;
    }
    const reco::TrackCollection tC2 = *TC2;

    // Step B: create empty output collection
    outputTrks = std::auto_ptr<reco::TrackCollection>(new reco::TrackCollection);
    refTrks = e.getRefBeforePut<reco::TrackCollection>();

    if (copyExtras_) {
        outputTrkExtras = std::auto_ptr<reco::TrackExtraCollection>(new reco::TrackExtraCollection);
	outputTrkExtras->reserve(TC1->size()+TC2->size());
        refTrkExtras    = e.getRefBeforePut<reco::TrackExtraCollection>();
        outputTrkHits   = std::auto_ptr<TrackingRecHitCollection>(new TrackingRecHitCollection);
	outputTrkHits->reserve((TC1->size()+TC2->size())*25);
        refTrkHits      = e.getRefBeforePut<TrackingRecHitCollection>();
	if (makeReKeyedSeeds_){
	  outputSeeds = std::auto_ptr<TrajectorySeedCollection>(new TrajectorySeedCollection);
	  outputSeeds->reserve(TC1->size()+TC2->size());
	  refTrajSeeds = e.getRefBeforePut<TrajectorySeedCollection>();
	}
    }

    outputTrajs = std::auto_ptr< std::vector<Trajectory> >(new std::vector<Trajectory>());
    outputTrajs->reserve(TC1->size()+TC2->size());
    outputTTAss = std::auto_ptr< TrajTrackAssociationCollection >(new TrajTrackAssociationCollection());
    //outputTTAss->reserve(TC1->size()+TC2->size());//how do I reserve space for an association map?

  //
  //  no input tracks
  //

//    if ( tC1.empty() ){
//      LogDebug("RoadSearch") << "Found " << output.size() << " clouds.";
//      e.put(output);
//      return;
//    }

  //
  //  quality cuts first
  //
    int i;

    std::vector<int> selected1; for (unsigned int i=0; i<tC1.size(); ++i){selected1.push_back(1);}

   if ( 0<tC1.size() ){
      i=-1;
      for (reco::TrackCollection::const_iterator track=tC1.begin(); track!=tC1.end(); track++){
        i++;
        if ((short unsigned)track->ndof() < 1){
          selected1[i]=0;
          //std::cout << "L1Track "<< i << " rejected in SimpleTrackListMerger; ndof() < 1" << std::endl ;
          continue;
        }
        if (track->normalizedChi2() > maxNormalizedChisq){
          selected1[i]=0;
          //std::cout << "L1Track "<< i << " rejected in SimpleTrackListMerger; normalizedChi2() > maxNormalizedChisq " << track->normalizedChi2() << " " << maxNormalizedChisq << std::endl ;
          continue;
        }
        if (track->found() < minFound){
          selected1[i]=0;
          //std::cout << "L1Track "<< i << " rejected in SimpleTrackListMerger; found() < minFound " << track->found() << " " << minFound << std::endl ;
          continue;
        }
        if (track->pt() < minPT){
          selected1[i]=0;
          //std::cout << "L1Track "<< i << " rejected in SimpleTrackListMerger; pt() < minPT " << track->pt() << " " << minPT << std::endl ;
          continue;
        }
      }//end loop over tracks
   }//end more than 0 track


    std::vector<int> selected2; for (unsigned int i=0; i<tC2.size(); ++i){selected2.push_back(1);}

   if ( 0<tC2.size() ){
      i=-1;
      for (reco::TrackCollection::const_iterator track=tC2.begin(); track!=tC2.end(); track++){
        i++;
        if ((short unsigned)track->ndof() < 1){
          selected2[i]=0;
          //std::cout << "L2Track "<< i << " rejected in SimpleTrackListMerger; ndof() < 1" << std::endl ;
          continue;
        }
        if (track->normalizedChi2() > maxNormalizedChisq){
          selected2[i]=0;
          //std::cout << "L2Track "<< i << " rejected in SimpleTrackListMerger; normalizedChi2() > maxNormalizedChisq " << track->normalizedChi2() << " " << maxNormalizedChisq << std::endl ;
          continue;
        }
        if (track->found() < minFound){
          selected2[i]=0;
          //std::cout << "L2Track "<< i << " rejected in SimpleTrackListMerger; found() < minFound " << track->found() << " " << minFound << std::endl ;
          continue;
        }
        if (track->pt() < minPT){
          selected2[i]=0;
          //std::cout << "L2Track "<< i << " rejected in SimpleTrackListMerger; pt() < minPT " << track->pt() << " " << minPT << std::endl ;
          continue;
        }
      }//end loop over tracks
   }//end more than 0 track

   std::map<reco::TrackCollection::const_iterator, std::vector<const TrackingRecHit*> > rh1;
   std::map<reco::TrackCollection::const_iterator, std::vector<const TrackingRecHit*> > rh2;
   for (reco::TrackCollection::const_iterator track=tC1.begin(); track!=tC1.end(); ++track){
     trackingRecHit_iterator itB = track->recHitsBegin();
     trackingRecHit_iterator itE = track->recHitsEnd();
     for (trackingRecHit_iterator it = itB;  it != itE; ++it) {
       const TrackingRecHit* hit = &(**it);
       rh1[track].push_back(hit);
     }
   }
   for (reco::TrackCollection::const_iterator track=tC2.begin(); track!=tC2.end(); ++track){
     trackingRecHit_iterator jtB = track->recHitsBegin();
     trackingRecHit_iterator jtE = track->recHitsEnd();
     for (trackingRecHit_iterator jt = jtB;  jt != jtE; ++jt) {
       const TrackingRecHit* hit = &(**jt);
       rh2[track].push_back(hit);
     }
   }

   if ( (0<tC1.size())&&(0<tC2.size()) ){
    i=-1;
    for (reco::TrackCollection::const_iterator track=tC1.begin(); track!=tC1.end(); ++track){
      i++;
      if (!selected1[i])continue;
      std::vector<const TrackingRecHit*>& iHits = rh1[track];
      unsigned nh1 = iHits.size();
      int qualityMaskT1 = track->qualityMask();
      int j=-1;
      for (reco::TrackCollection::const_iterator track2=tC2.begin(); track2!=tC2.end(); ++track2){
        j++;
        if ((!selected2[j])||(!selected1[i]))continue;
	std::vector<const TrackingRecHit*>& jHits = rh2[track2];
	unsigned nh2 = jHits.size();
        int noverlap=0;
        int firstoverlap=0;
	for ( unsigned ih=0; ih<nh1; ++ih ) {
	  const TrackingRecHit* it = iHits[ih];
          if (it->isValid()){
            int jj=-1;
	    for ( unsigned jh=0; jh<nh2; ++jh ) {
	      const TrackingRecHit* jt = jHits[jh];
              jj++;
	      if (jt->isValid()){
               if (!use_sharesInput){
                float delta = fabs ( it->localPosition().x()-jt->localPosition().x() );
                if ((it->geographicalId()==jt->geographicalId())&&(delta<epsilon)) {
		  noverlap++;
		  if ( allowFirstHitShare && ( ih == 0 ) && ( jh == 0 ) ) firstoverlap=1;
		}
               }else{
		if ( it->sharesInput(jt,TrackingRecHit::some) ) {
		  noverlap++;
		  if ( allowFirstHitShare && ( ih == 0 ) && ( jh == 0 ) ) firstoverlap=1;
		}
	       }
	      }
            }
          }
        }
	int newQualityMask = (qualityMaskT1 | track2->qualityMask()); // take OR of trackQuality
	int nhit1 = track->numberOfValidHits();
	int nhit2 = track2->numberOfValidHits();
        if ( (noverlap-firstoverlap) > (std::min(nhit1,nhit2)-firstoverlap)*shareFrac ) {
	  double score1 = foundHitBonus*nhit1 - lostHitPenalty*track->numberOfLostHits() - track->chi2();
	  double score2 = foundHitBonus*nhit2 - lostHitPenalty*track2->numberOfLostHits() - track2->chi2();
	  const double almostSame = 1.001;
          if ( score1 > almostSame * score2 ){
            selected2[j]=0;
	    selected1[i]=10+newQualityMask; // add 10 to avoid the case where mask = 1
          }else if ( score2 > almostSame * score1 ){
              selected1[i]=0;
	      selected2[j]=10+newQualityMask;  // add 10 to avoid the case where mask = 1
	  }else{
	    if (track->algo() <= track2->algo()) {
	      selected2[j]=0;
	      selected1[i]=10+newQualityMask; // add 10 to avoid the case where mask = 1
	    }else{
	      selected1[i]=0;
	      selected2[j]=10+newQualityMask; // add 10 to avoid the case where mask = 1
            }
          }
        }//end got a duplicate
      }//end track2 loop
    }//end track loop
   }//end more than 1 track

  //
  //  output selected tracks - if any
  //
   trackRefs.resize(tC1.size()+tC2.size());
   std::vector<edm::RefToBase<TrajectorySeed> > seedsRefs(tC1.size()+tC2.size());
   size_t current = 0;

   if ( 0<tC1.size() ){
     i=0;
     for (reco::TrackCollection::const_iterator track=tC1.begin(); track!=tC1.end();
	  ++track, ++current, ++i){
      if (!selected1[i]){
	trackRefs[current] = reco::TrackRef();
	continue;
      }
      const reco::Track & theTrack = * track;
      //fill the TrackCollection
      outputTrks->push_back( reco::Track( theTrack ) );
      if (selected1[i]>1 && promoteQuality){
	outputTrks->back().setQualityMask(selected1[i]-10);
	outputTrks->back().setQuality(qualityToSet);
      }
      if (copyExtras_) {
	  //--------NEW----------
	  edm::RefToBase<TrajectorySeed> origSeedRef = theTrack.seedRef();
	  //creating a seed with rekeyed clusters if required
	  if (makeReKeyedSeeds_){
	    bool doRekeyOnThisSeed=false;

	    edm::InputTag clusterRemovalInfos("");
	    //grab on of the hits of the seed
	    if (origSeedRef->nHits()!=0){
	      TrajectorySeed::const_iterator firstHit=origSeedRef->recHits().first;
	      const TrackingRecHit *hit = &*firstHit;
	      if (firstHit->isValid()){
		edm::ProductID  pID=clusterProduct(hit);
		// the cluster collection either produced a removalInfo or mot
		//get the clusterremoval info from the provenance: will rekey if this is found
		edm::Handle<reco::ClusterRemovalInfo> CRIh;
		edm::Provenance prov=e.getProvenance(pID);
		clusterRemovalInfos=edm::InputTag(prov.moduleLabel(),
						  prov.productInstanceName(),
						  prov.processName());
		// In case this is switched on, will have to revist that, and declare mayConsumes...
		doRekeyOnThisSeed=e.getByLabel(clusterRemovalInfos,CRIh);
	      }//valid hit
	    }//nhit!=0

	    if (doRekeyOnThisSeed && !(clusterRemovalInfos==edm::InputTag("")))
	      {
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
	  //--------NEW----------
          // Fill TrackExtra collection
	  outputTrkExtras->push_back( reco::TrackExtra(
                        theTrack.outerPosition(), theTrack.outerMomentum(), theTrack.outerOk(),
                        theTrack.innerPosition(), theTrack.innerMomentum(), theTrack.innerOk(),
                        theTrack.outerStateCovariance(), theTrack.outerDetId(),
                        theTrack.innerStateCovariance(), theTrack.innerDetId(),
                        theTrack.seedDirection(), origSeedRef ) );
	  seedsRefs[current]=origSeedRef;
          outputTrks->back().setExtra( reco::TrackExtraRef( refTrkExtras, outputTrkExtras->size() - 1) );
          reco::TrackExtra & tx = outputTrkExtras->back();
	  tx.setResiduals(theTrack.residuals());
          // fill TrackingRecHits
          std::vector<const TrackingRecHit*>& iHits = rh1[track];
          unsigned nh1 = iHits.size();
          for ( unsigned ih=0; ih<nh1; ++ih ) {
            const TrackingRecHit* hit = iHits[ih];
            //for( trackingRecHit_iterator hit = itB; hit != itE; ++hit ) {
            outputTrkHits->push_back( hit->clone() );
            tx.add( TrackingRecHitRef( refTrkHits, outputTrkHits->size() - 1) );
          }
      }
      trackRefs[current] = reco::TrackRef(refTrks, outputTrks->size() - 1);


    }//end faux loop over tracks
   }//end more than 0 track

   //Fill the trajectories, etc. for 1st collection
   edm::Handle< std::vector<Trajectory> > hTraj1;
   e.getByToken(trackProducer1TrajToken, hTraj1);
   edm::Handle< TrajTrackAssociationCollection > hTTAss1;
   e.getByToken(trackProducer1AssToken, hTTAss1);
   refTrajs    = e.getRefBeforePut< std::vector<Trajectory> >();

   if (!hTraj1.failedToGet() && !hTTAss1.failedToGet()){
   for (size_t i = 0, n = hTraj1->size(); i < n; ++i) {
     edm::Ref< std::vector<Trajectory> > trajRef(hTraj1, i);
     TrajTrackAssociationCollection::const_iterator match = hTTAss1->find(trajRef);
     if (match != hTTAss1->end()) {
       const edm::Ref<reco::TrackCollection> &trkRef = match->val;
       short oldKey = static_cast<short>(trkRef.key());
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

   short offset = current; //save offset into trackRefs

   if ( 0<tC2.size() ){
    i=0;
    for (reco::TrackCollection::const_iterator track=tC2.begin(); track!=tC2.end();
	 ++track, ++current, ++i){
      if (!selected2[i]){
	trackRefs[current] = reco::TrackRef();
	continue;
      }
      const reco::Track & theTrack = * track;
      //fill the TrackCollection
      outputTrks->push_back( reco::Track( theTrack ) );
      if (selected2[i]>1 && promoteQuality){
	outputTrks->back().setQualityMask(selected2[i]-10);
	outputTrks->back().setQuality(qualityToSet);
      }
      if (copyExtras_) {
	  //--------NEW----------
	  edm::RefToBase<TrajectorySeed> origSeedRef = theTrack.seedRef();
	  //creating a seed with rekeyed clusters if required
	  if (makeReKeyedSeeds_){
	    bool doRekeyOnThisSeed=false;

	    edm::InputTag clusterRemovalInfos("");
	    //grab on of the hits of the seed
	    if (origSeedRef->nHits()!=0){
	      TrajectorySeed::const_iterator firstHit=origSeedRef->recHits().first;
	      const TrackingRecHit *hit = &*firstHit;
	      if (firstHit->isValid()){
		edm::ProductID  pID=clusterProduct(hit);
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

	    if (doRekeyOnThisSeed && !(clusterRemovalInfos==edm::InputTag("")))
	      {
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
	      }//doRekeyOnThisSeed=true
	      else{
		//just copy the one we had before
		outputSeeds->push_back( TrajectorySeed(*origSeedRef));
	      }
	    edm::Ref<TrajectorySeedCollection> pureRef(refTrajSeeds, outputSeeds->size()-1);
	    origSeedRef=edm::RefToBase<TrajectorySeed>( pureRef);
	  }//creating a new seed and rekeying it rechit clusters.
	  //--------NEW----------
          // Fill TrackExtra collection
          outputTrkExtras->push_back( reco::TrackExtra(
                        theTrack.outerPosition(), theTrack.outerMomentum(), theTrack.outerOk(),
                        theTrack.innerPosition(), theTrack.innerMomentum(), theTrack.innerOk(),
                        theTrack.outerStateCovariance(), theTrack.outerDetId(),
                        theTrack.innerStateCovariance(), theTrack.innerDetId(),
                        theTrack.seedDirection(), origSeedRef ) );
	  seedsRefs[current]=origSeedRef;
          outputTrks->back().setExtra( reco::TrackExtraRef( refTrkExtras, outputTrkExtras->size() - 1) );
          reco::TrackExtra & tx = outputTrkExtras->back();
	  tx.setResiduals(theTrack.residuals());
          // fill TrackingRecHits
          std::vector<const TrackingRecHit*>& jHits = rh2[track];
          unsigned nh2 = jHits.size();
          for ( unsigned jh=0; jh<nh2; ++jh ) {
            const TrackingRecHit* hit = jHits[jh];
            outputTrkHits->push_back( hit->clone() );
            tx.add( TrackingRecHitRef( refTrkHits, outputTrkHits->size() - 1) );
          }
      }
      trackRefs[current] = reco::TrackRef(refTrks, outputTrks->size() - 1);

    }//end faux loop over tracks
   }//end more than 0 track

   //Fill the trajectories, etc. for 2nd collection
   edm::Handle< std::vector<Trajectory> > hTraj2;
   e.getByToken(trackProducer2TrajToken, hTraj2);
   edm::Handle< TrajTrackAssociationCollection > hTTAss2;
   e.getByToken(trackProducer2AssToken, hTTAss2);

   if (!hTraj2.failedToGet() && !hTTAss2.failedToGet()){
   for (size_t i = 0, n = hTraj2->size(); i < n; ++i) {
     edm::Ref< std::vector<Trajectory> > trajRef(hTraj2, i);
     TrajTrackAssociationCollection::const_iterator match = hTTAss2->find(trajRef);
     if (match != hTTAss2->end()) {
       const edm::Ref<reco::TrackCollection> &trkRef = match->val;
       short oldKey = static_cast<short>(trkRef.key()) + offset;
       if (trackRefs[oldKey].isNonnull()) {
           outputTrajs->push_back( Trajectory(*trajRef) );
	   //if making extras and the seeds at the same time, change the seed ref on the trajectory
	   if (copyExtras_ && makeReKeyedSeeds_)
	     outputTrajs->back().setSeedRef( seedsRefs[oldKey] );
	   outputTTAss->insert ( edm::Ref< std::vector<Trajectory> >(refTrajs, outputTrajs->size() - 1),
                   trackRefs[oldKey] );
       }
     }
   }}

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
