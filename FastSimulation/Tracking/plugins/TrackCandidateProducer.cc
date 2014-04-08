#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h" 
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2DCollection.h" 
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "FastSimulation/Tracking/interface/TrackerRecHit.h"
//#include "FastSimulation/Tracking/interface/TrackerRecHitSplit.h"

#include "FastSimulation/Tracking/plugins/TrackCandidateProducer.h"

#include <vector>
#include <map>

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

//Propagator withMaterial
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"


//

//for debug only 
//#define FAMOS_DEBUG

TrackCandidateProducer::TrackCandidateProducer(const edm::ParameterSet& conf):thePropagator(0) 
{  
#ifdef FAMOS_DEBUG
  std::cout << "TrackCandidateProducer created" << std::endl;
#endif

  // The main product is a track candidate collection.
  produces<TrackCandidateCollection>();

  // These products contain tracks already reconstructed at this level
  // (No need to reconstruct them twice!)
  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
  produces<std::vector<Trajectory> >();
  produces<TrajTrackAssociationCollection>();
  
  // The name of the seed producer
  seedProducer = conf.getParameter<edm::InputTag>("SeedProducer");

  // The name of the recHit producer
  hitProducer = conf.getParameter<edm::InputTag>("HitProducer");

  // The name of the track producer (tracks already produced need not be produced again!)
  // trackProducer = conf.getParameter<edm::InputTag>("TrackProducer");
  trackProducers = conf.getParameter<std::vector<edm::InputTag> >("TrackProducers");

  // Copy (or not) the tracks already produced in a new collection
  keepFittedTracks = conf.getParameter<bool>("KeepFittedTracks");

  // The minimum number of crossed layers
  minNumberOfCrossedLayers = conf.getParameter<unsigned int>("MinNumberOfCrossedLayers");

  // The maximum number of crossed layers
  maxNumberOfCrossedLayers = conf.getParameter<unsigned int>("MaxNumberOfCrossedLayers");

  // Reject overlapping hits?
  rejectOverlaps = conf.getParameter<bool>("OverlapCleaning");

  // Split hits ?
  splitHits = conf.getParameter<bool>("SplitHits");

  // Reject tracks with several seeds ?
  // Typically don't do that at HLT for electrons, but do it otherwise
  seedCleaning = conf.getParameter<bool>("SeedCleaning");

  simTracks_ = conf.getParameter<edm::InputTag>("SimTracks");
  estimatorCut_= conf.getParameter<double>("EstimatorCut");

  // consumes
  seedToken = consumes<edm::View<TrajectorySeed> >(seedProducer);
  recHitToken = consumes<SiTrackerGSMatchedRecHit2DCollection>(hitProducer);
  edm::InputTag _label("famosSimHits");
  simVertexToken = consumes<edm::SimVertexContainer>(_label);
  simTrackToken = consumes<edm::SimTrackContainer>(_label);
  for(unsigned tprod=0; tprod < trackProducers.size(); ++tprod){
    trackTokens.push_back(consumes<reco::TrackCollection>(trackProducers[tprod]));
    trajectoryTokens.push_back(consumes<std::vector<Trajectory> >(trackProducers[tprod]));
    assoMapTokens.push_back(consumes<TrajTrackAssociationCollection>(trackProducers[tprod]));
  }
}

  
// Virtual destructor needed.
TrackCandidateProducer::~TrackCandidateProducer() {

  if(thePropagator) delete thePropagator;

  // do nothing
#ifdef FAMOS_DEBUG
  std::cout << "TrackCandidateProducer destructed" << std::endl;
#endif


} 
 
void 
TrackCandidateProducer::beginRun(edm::Run const&, const edm::EventSetup & es) {

  //services

  edm::ESHandle<MagneticField>          magField;
  edm::ESHandle<TrackerGeometry>        geometry;

  es.get<IdealMagneticFieldRecord>().get(magField);
  es.get<TrackerDigiGeometryRecord>().get(geometry);

  theMagField = &(*magField);
  theGeometry = &(*geometry);

  thePropagator = new PropagatorWithMaterial(alongMomentum,0.105,&(*theMagField)); 
}
  
  // Functions that gets called by framework every event
void 
TrackCandidateProducer::produce(edm::Event& e, const edm::EventSetup& es) {        

#ifdef FAMOS_DEBUG
  std::cout << "################################################################" << std::endl;
  std::cout << " TrackCandidateProducer produce init " << std::endl;
#endif

  // Useful typedef's to avoid retyping
  typedef std::pair<reco::TrackRef,edm::Ref<std::vector<Trajectory> > > TrackPair;
  typedef std::map<unsigned,TrackPair> TrackMap;

  // The produced objects
  std::auto_ptr<TrackCandidateCollection> output(new TrackCandidateCollection);    
  std::auto_ptr<reco::TrackCollection> recoTracks(new reco::TrackCollection);    
  std::auto_ptr<TrackingRecHitCollection> recoHits(new TrackingRecHitCollection);
  std::auto_ptr<reco::TrackExtraCollection> recoTrackExtras(new reco::TrackExtraCollection);
  std::auto_ptr<std::vector<Trajectory> > recoTrajectories(new std::vector<Trajectory> );
  std::auto_ptr<TrajTrackAssociationCollection> recoTrajTrackMap( new TrajTrackAssociationCollection() );
  
  // Get the seeds
  // edm::Handle<TrajectorySeedCollection> theSeeds;
  edm::Handle<edm::View<TrajectorySeed> > theSeeds;
  e.getByToken(seedToken,theSeeds);

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  es.get<IdealGeometryRecord>().get(tTopoHand);
  const TrackerTopology *tTopo=tTopoHand.product();


  // No seed -> output an empty track collection
  if(theSeeds->size() == 0) {
    e.put(output);
    e.put(recoTracks);
    e.put(recoHits);
    e.put(recoTrackExtras);
    e.put(recoTrajectories);
    e.put(recoTrajTrackMap);
    return;
  }

  // Get the GS RecHits
  //  edm::Handle<SiTrackerGSRecHit2DCollection> theGSRecHits;
  edm::Handle<SiTrackerGSMatchedRecHit2DCollection> theGSRecHits;
  e.getByToken(recHitToken, theGSRecHits);

  //get other general things
  const std::vector<unsigned> theSimTrackIds = theGSRecHits->ids(); 
  // SimTracks and SimVertices

  edm::Handle<edm::SimVertexContainer> theSimVtx;
  e.getByToken(simVertexToken,theSimVtx);
  edm::Handle<edm::SimTrackContainer> theSTC;
  e.getByToken(simTrackToken,theSTC);

  const edm::SimTrackContainer* theSimTracks = &(*theSTC);
  LogDebug("FastTracking")<<"looking at: "<< theSimTrackIds.size()<<" simtracks.";
 

     

  // The input track collection + extra's
  /*
  edm::Handle<reco::TrackCollection> theTrackCollection;
  edm:: Handle<std::vector<Trajectory> > theTrajectoryCollection;
  edm::Handle<TrajTrackAssociationCollection> theAssoMap;  
  bool isTrackCollection = e.getByLabel(trackProducer,theTrackCollection);
  */
  std::vector<edm::Handle<reco::TrackCollection> > theTrackCollections;
  std::vector<edm:: Handle<std::vector<Trajectory> > > theTrajectoryCollections;
  std::vector<edm::Handle<TrajTrackAssociationCollection> > theAssoMaps;
  std::vector<bool> isTrackCollections;
  TrajTrackAssociationCollection::const_iterator anAssociation;  
  TrajTrackAssociationCollection::const_iterator lastAssociation;
  TrackMap theTrackMap;
  unsigned nCollections = trackProducers.size();
  unsigned nRecoHits = 0;

  if ( nCollections ) { 
    theTrackCollections.resize(nCollections);
    theTrajectoryCollections.resize(nCollections);
    theAssoMaps.resize(nCollections);
    isTrackCollections.resize(nCollections);
    for ( unsigned tprod=0; tprod < nCollections; ++tprod ) { 
      isTrackCollections[tprod] = e.getByToken(trackTokens[tprod],theTrackCollections[tprod]); 

      if ( isTrackCollections[tprod] ) { 
	// The track collection
	reco::TrackCollection::const_iterator aTrack = theTrackCollections[tprod]->begin();
	reco::TrackCollection::const_iterator lastTrack = theTrackCollections[tprod]->end();
	// The numbers of hits
	for ( ; aTrack!=lastTrack; ++aTrack ) nRecoHits+= aTrack->recHitsSize();
	e.getByToken(trajectoryTokens[tprod],theTrajectoryCollections[tprod]);
	e.getByToken(assoMapTokens[tprod],theAssoMaps[tprod]);
	// The association between trajectories and tracks
	anAssociation = theAssoMaps[tprod]->begin();
	lastAssociation = theAssoMaps[tprod]->end(); 
#ifdef FAMOS_DEBUG
	std::cout << "Input Track Producer " << tprod << " : " << trackProducers[tprod] << std::endl;
	std::cout << "List of tracks already reconstructed " << std::endl;
#endif
	// Build the map of correspondance between reco tracks and sim tracks
	for ( ; anAssociation != lastAssociation; ++anAssociation ) { 
	  edm::Ref<std::vector<Trajectory> > aTrajectoryRef = anAssociation->key;
	  reco::TrackRef aTrackRef = anAssociation->val;
	  // Find the simtrack id of the reconstructed track
	  int recoTrackId = findId(*aTrackRef);
	  if ( recoTrackId < 0 ) continue;
#ifdef FAMOS_DEBUG
	  std::cout << recoTrackId << " ";
#endif
	  // And store it.
	  theTrackMap[recoTrackId] = TrackPair(aTrackRef,aTrajectoryRef);
	}
#ifdef FAMOS_DEBUG
	std::cout << std::endl;
#endif
      }
    }
    // This is to save some time at push_back.
    recoHits->reserve(nRecoHits); 
  }


  // Loop over the seeds
  int currentTrackId = -1;
  /*
  TrajectorySeedCollection::const_iterator aSeed = theSeeds->begin();
  TrajectorySeedCollection::const_iterator lastSeed = theSeeds->end();
  for ( ; aSeed!=lastSeed; ++aSeed ) { 
    // The first hit of the seed  and its simtrack id
  */
  /* */
#ifdef FAMOS_DEBUG
  std::cout << "Input seed Producer : " << seedProducer << std::endl;
  std::cout << "Number of seeds : " << theSeeds->size() << std::endl;
#endif
  unsigned seed_size = theSeeds->size(); 
  for (unsigned seednr = 0; seednr < seed_size; ++seednr){
    
    LogDebug("FastTracking")<<"looking at seed #:"<<seednr;

    // The seed
    const BasicTrajectorySeed* aSeed = &((*theSeeds)[seednr]);

    std::vector<int> simTrackIds;
    std::map<int,TrajectoryStateOnSurface> seedStates;
    std::map<int,TrajectoryStateOnSurface> simtkStates;

    TrackerRecHit theFirstSeedingTrackerRecHit;
    if (theSeeds->at(seednr).nHits()==0){
      //new stuff for no hits on seed

      LogDebug("FastTracking")<<" seed with no hits to be considered.";

      //moved out of the loop
      //edm::ESHandle<MagneticField> field;
      //es.get<IdealMagneticFieldRecord>().get(field);

      PTrajectoryStateOnDet ptod =theSeeds->at(seednr).startingState();
      DetId id(ptod.detId());
      const GeomDet * g = theGeometry->idToDet(id);
      const Surface * surface=&g->surface();
      
      TrajectoryStateOnSurface seedState(trajectoryStateTransform::transientState(ptod,surface,theMagField));
      
      edm::ESHandle<Propagator> propagator;
      es.get<TrackingComponentsRecord>().get("AnyDirectionAnalyticalPropagator",propagator);
      
      //moved out of the loop 
      //      const std::vector<unsigned> theSimTrackIds = theGSRecHits->ids(); 
      //      edm::Handle<edm::SimTrackContainer> theSTC;
      //      e.getByLabel(simTracks_,theSTC);
      //      const edm::SimTrackContainer* theSimTracks = &(*theSTC);

      double minimunEst=1000000;
      LogDebug("FastTracking")<<"looking at: "<< theSimTrackIds.size()<<" simtracks.";
      for ( unsigned tkId=0;  tkId != theSimTrackIds.size(); ++tkId ) {
	
	const SimTrack & simtrack = (*theSimTracks)[theSimTrackIds[tkId]];

	GlobalPoint position(simtrack.trackerSurfacePosition().x(),
			     simtrack.trackerSurfacePosition().y(),
			     simtrack.trackerSurfacePosition().z());
	
	GlobalVector momentum(simtrack.trackerSurfaceMomentum().x(),
			      simtrack.trackerSurfaceMomentum().y(),
			      simtrack.trackerSurfaceMomentum().z());

	if (position.basicVector().dot( momentum.basicVector() ) * seedState.globalPosition().basicVector().dot(seedState.globalMomentum().basicVector()) <0. ){
	  LogDebug("FastTracking")<<"not on the same direction.";
	  continue;
	}

	//no charge mis-identification ... FIXME
	int charge = (int) simtrack.charge();
	GlobalTrajectoryParameters glb_parameters(position,
						  momentum,
						  charge,
						  theMagField);
	FreeTrajectoryState simtrack_trackerstate(glb_parameters);
	
	TrajectoryStateOnSurface simtrack_comparestate = propagator->propagate(simtrack_trackerstate,*surface);

	  
	if (!simtrack_comparestate.isValid()){
	  LogDebug("FastTracking")<<" ok this is a state-based seed. simtrack state does not propagate to the seed surface. skipping.";
	  continue;}
	
	if (simtrack_comparestate.globalPosition().basicVector().dot(simtrack_comparestate.globalMomentum().basicVector()) * seedState.globalPosition().basicVector().dot(seedState.globalMomentum().basicVector()) <0. ){
	  LogDebug("FastTracking")<<"not on the same direction.";
	  continue;
	}

	
	AlgebraicVector5 v(seedState.localParameters().vector() - simtrack_comparestate.localParameters().vector());
	AlgebraicSymMatrix55 m(seedState.localError().matrix());
	bool ierr = !m.Invert();
	if ( ierr ){
	  edm::LogWarning("FastTracking") <<" Candidate Producer cannot invert the error matrix! - Skipping...";
	  continue;
	}
	double est = ROOT::Math::Similarity(v,m);
      	LogDebug("FastTracking")<<"comparing two state on the seed surface:\n"
					  <<"seed: "<<seedState
					  <<"sim: "<<simtrack_comparestate
					  <<"\n estimator is: "<<est;

	if (est<minimunEst)	  minimunEst=est;
	if (est<estimatorCut_){
	  simTrackIds.push_back(theSimTrackIds[tkId]);
	  //making a state with exactly the sim track parameters
	  //the initial errors are set to unity just for kicks
	  //	  AlgebraicSymMatrix C(5,1);// C*=50;
	  //new attempt!!!!
	  AlgebraicSymMatrix55 C = seedState.curvilinearError().matrix();
	  C *= 0.0000001;

	  seedStates[theSimTrackIds[tkId]] = TrajectoryStateOnSurface(simtrack_comparestate.globalParameters(),
								      CurvilinearTrajectoryError(C),
								      seedState.surface());
	  LogDebug("FastTracking")<<"the compatibility estimator is: "<<est<<" for track id: "<<simTrackIds.back();
	}
      }//SimTrack loop
      if (simTrackIds.size()==0) LogDebug("FastTracking")<<"could not find any simtrack within errors, closest was at: "<<minimunEst;
    }//seed has 0 hit.
    else{
      //same old stuff
      // Find the first hit of the Seed
      TrajectorySeed::range theSeedingRecHitRange = aSeed->recHits();
      const SiTrackerGSMatchedRecHit2D * theFirstSeedingRecHit = (const SiTrackerGSMatchedRecHit2D*) (&(*(theSeedingRecHitRange.first)));
      theFirstSeedingTrackerRecHit = TrackerRecHit(theFirstSeedingRecHit,theGeometry,tTopo);
      // The SimTrack id associated to that recHit
      simTrackIds.push_back( theFirstSeedingRecHit->simtrackId() );
    }

    //from then on, only the simtrack IDs are usefull.
    //now loop over all possible trackid for this seed.
    //an actual seed can be shared by two tracks in dense envirronement, and also for hit-less seeds.
    for (unsigned int iToMake=0;iToMake!=simTrackIds.size();++iToMake){
      int simTrackId = simTrackIds[iToMake];
      
      // Don't consider seeds belonging to a track already considered 
      // (Equivalent to seed cleaning)
      if ( seedCleaning && currentTrackId == simTrackId ) continue;
      currentTrackId = simTrackId;
      
      // A vector of TrackerRecHits belonging to the track and the number of crossed layers
      std::vector<TrackerRecHit> theTrackerRecHits;
      unsigned theNumberOfCrossedLayers = 0;
      
      // The track has indeed been reconstructed already -> Save the pertaining info
      TrackMap::const_iterator theTrackIt = theTrackMap.find(simTrackId);
      if ( nCollections && theTrackIt != theTrackMap.end() ) { 
	
	if ( keepFittedTracks ) { 
	  LogDebug("FastTracking") << "Track " << simTrackId << " already reconstructed -> copy it";
	  // The track and trajectroy references
	  reco::TrackRef aTrackRef = theTrackIt->second.first;
	  edm::Ref<std::vector<Trajectory> > aTrajectoryRef = theTrackIt->second.second;
	  
	  // A copy of the track
	  reco::Track aRecoTrack(*aTrackRef);
	  recoTracks->push_back(aRecoTrack);      
	  
	  // A copy of the hits
	  unsigned nh = aRecoTrack.recHitsSize();
	  for ( unsigned ih=0; ih<nh; ++ih ) {
	    TrackingRecHit *hit = aRecoTrack.recHit(ih)->clone();
	    recoHits->push_back(hit);
	  }
	  
	  // A copy of the trajectories
	  recoTrajectories->push_back(*aTrajectoryRef);
	  
	}// keepFitterTracks
	else {	  
	  LogDebug("FastTracking") << "Track " << simTrackId << " already reconstructed -> ignore it";
	}

	// The track was not saved -> create a track candidate.
	
      } //already existing collection of tracks
      else{//no collection of tracks already exists

	LogDebug("FastTracking")<<"Track " << simTrackId << " is considered to return a track candidate" ;

	// Get all the rechits associated to this track
	SiTrackerGSMatchedRecHit2DCollection::range theRecHitRange = theGSRecHits->get(simTrackId);
	SiTrackerGSMatchedRecHit2DCollection::const_iterator theRecHitRangeIteratorBegin = theRecHitRange.first;
	SiTrackerGSMatchedRecHit2DCollection::const_iterator theRecHitRangeIteratorEnd   = theRecHitRange.second;
	SiTrackerGSMatchedRecHit2DCollection::const_iterator iterRecHit;

	LogDebug("FastTracking")<<"counting: "<<theRecHitRangeIteratorEnd-theRecHitRangeIteratorBegin<<" hits to be considered.";

	bool firstRecHit = true;
	TrackerRecHit theCurrentRecHit, thePreviousRecHit;
	TrackerRecHit theFirstHitComp, theSecondHitComp;
	
	for ( iterRecHit = theRecHitRangeIteratorBegin; 
	      iterRecHit != theRecHitRangeIteratorEnd; 
	      ++iterRecHit) {
	  
	  // Check the number of crossed layers
	  if ( theNumberOfCrossedLayers >= maxNumberOfCrossedLayers ) continue;
	  
	  // Get current and previous rechits
	  if(!firstRecHit) thePreviousRecHit = theCurrentRecHit;
	  theCurrentRecHit = TrackerRecHit(&(*iterRecHit),theGeometry,tTopo);
	  
	  //>>>>>>>>>BACKBUILDING CHANGE: DO NOT STAT FROM THE FIRST HIT OF THE SEED

	  // NOTE: checking the direction --> specific for OIHit only
	  //	  if( aSeed->direction()!=oppositeToMomentum ) { 
	  //  // Check that the first rechit is indeed the first seeding hit
	  //  if ( firstRecHit && theCurrentRecHit != theFirstSeedingTrackerRecHit && theSeeds->at(seednr).nHits()!=0 ) continue;
	  // }
	  //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

	  // Count the number of crossed layers
	  if ( !theCurrentRecHit.isOnTheSameLayer(thePreviousRecHit) ) 
	    ++theNumberOfCrossedLayers;
	  
	  // Add all rechits (Grouped Trajectory Builder) from this hit onwards
	  // Always add the first seeding rechit anyway
	  if ( !rejectOverlaps || firstRecHit ) {  
	    // Split matched hits (if requested / possible )
	    if ( splitHits && theCurrentRecHit.matchedHit()->isMatched() ) addSplitHits(theCurrentRecHit,theTrackerRecHits);
	    else theTrackerRecHits.push_back(theCurrentRecHit);	      // No splitting   
	    firstRecHit = false;
	    
	    // And now treat the following RecHits if hits in the same layer 
	    // have to be rejected - The split option is not 
	  } else { 
	    
	    // Not the same layer : Add the current hit
	    if ( theCurrentRecHit.subDetId()    != thePreviousRecHit.subDetId() || 
		 theCurrentRecHit.layerNumber() != thePreviousRecHit.layerNumber() ) {
	      
	      // Split matched hits (if requested / possible )
	      if ( splitHits && theCurrentRecHit.matchedHit()->isMatched() ) addSplitHits(theCurrentRecHit,theTrackerRecHits);
	      else 		theTrackerRecHits.push_back(theCurrentRecHit); 		// No splitting   	      
	      
	      // Same layer : keep the current hit if better, and drop the other - otherwise do nothing  
	    } else if ( theCurrentRecHit.localError() < thePreviousRecHit.localError() ) { 
	      
	      // Split matched hits (if requested / possible )
	      if( splitHits && theCurrentRecHit.matchedHit()->isMatched() ){
		// Remove the previous hit(s)
		theTrackerRecHits.pop_back();
		if ( thePreviousRecHit.matchedHit()->isMatched() ) theTrackerRecHits.pop_back();
		
		// Replace by the new hits
		addSplitHits(theCurrentRecHit,theTrackerRecHits);
	      }
	      // No splitting   
	      else {
		theTrackerRecHits.back() = theCurrentRecHit; // Replace the previous hit by the current hit
	      }
	      
	    } else {
	      
	      //keep the older rechit as a reference if the error of the new one is worse
	      theCurrentRecHit = thePreviousRecHit;
	    }  
	  }
	}
	
	// End of loop over the track rechits
      }//no collection of track already existed. adding the hits by hand.
    
      LogDebug("FastTracking")<<" number of hits: " << theTrackerRecHits.size()<<" after counting overlaps and splitting.";

      // 1) Create the OwnWector of TrackingRecHits
      edm::OwnVector<TrackingRecHit> recHits;
      unsigned nTrackerHits = theTrackerRecHits.size();
      recHits.reserve(nTrackerHits); // To save some time at push_back

      if (aSeed->direction()==oppositeToMomentum){
	LogDebug("FastTracking")<<"reversing the order of the hits";
	std::reverse(theTrackerRecHits.begin(),theTrackerRecHits.end());
      }

      for ( unsigned ih=0; ih<nTrackerHits; ++ih ) {
	TrackingRecHit* aTrackingRecHit = theTrackerRecHits[ih].hit()->clone();
	recHits.push_back(aTrackingRecHit);
	
	const DetId& detId = theTrackerRecHits[ih].hit()->geographicalId();
	LogDebug("FastTracking")
	  << "Added RecHit from detid " << detId.rawId() 
	  << " subdet = " << theTrackerRecHits[ih].subDetId() 
	  << " layer = " << theTrackerRecHits[ih].layerNumber()
	  << " ring = " << theTrackerRecHits[ih].ringNumber()
	  << " error = " << theTrackerRecHits[ih].localError()
	  << std::endl
	  
	  << "Track/z/r : "
	  << simTrackId << " " 
	  << theTrackerRecHits[ih].globalPosition().z() << " " 
	  << theTrackerRecHits[ih].globalPosition().perp() << std::endl;
	if ( theTrackerRecHits[ih].matchedHit() && theTrackerRecHits[ih].matchedHit()->isMatched() ) 
	  LogTrace("FastTracking") << "Matched : " << theTrackerRecHits[ih].matchedHit()->isMatched() 
					     << "Rphi Hit = " <<  theTrackerRecHits[ih].matchedHit()->monoHit()->simhitId()		 
					     << "Stereo Hit = " <<  theTrackerRecHits[ih].matchedHit()->stereoHit()->simhitId()
					     <<std::endl;
      }//loop over the rechits

    // Check the number of crossed layers
    if ( theNumberOfCrossedLayers < minNumberOfCrossedLayers ) {
      LogDebug("FastTracking")<<"not enough layer crossed ("<<theNumberOfCrossedLayers<<")";
      continue;
    }


    //>>>>>>>>>BACKBUILDING CHANGE: REPLACE THE STARTING STATE

    // Create a track Candidate (now with the reference to the seed!) .
    //PTrajectoryStateOnDet PTSOD = aSeed->startingState();
    PTrajectoryStateOnDet PTSOD;

    if (aSeed->nHits()==0){
      //stabilize the fit with the true simtrack state
      //in case of zero hits
      
      PTSOD = trajectoryStateTransform::persistentState(seedStates[simTrackId],aSeed->startingState().detId());
 
    } else {
      //create the initial state from the SimTrack
      int vertexIndex = (*theSimTracks)[currentTrackId].vertIndex();
      //   a) origin vertex
      GlobalPoint  position((*theSimVtx)[vertexIndex].position().x(),
			    (*theSimVtx)[vertexIndex].position().y(),
			    (*theSimVtx)[vertexIndex].position().z());
      
      //   b) initial momentum
      GlobalVector momentum( (*theSimTracks)[currentTrackId].momentum().x() , 
			     (*theSimTracks)[currentTrackId].momentum().y() , 
			     (*theSimTracks)[currentTrackId].momentum().z() );
      //   c) electric charge
      float        charge   = (*theSimTracks)[simTrackId].charge();
      //  -> inital parameters
      GlobalTrajectoryParameters initialParams(position,momentum,(int)charge,theMagField);
 //  -> large initial errors
      AlgebraicSymMatrix55 errorMatrix= AlgebraicMatrixID();    
      CurvilinearTrajectoryError initialError(errorMatrix);
      // -> initial state
      FreeTrajectoryState initialFTS(initialParams, initialError);      
#ifdef FAMOS_DEBUG
      std::cout << "TrajectorySeedProducer: FTS momentum " << initialFTS.momentum() << std::endl;
#endif
      const GeomDet* initialLayer = theGeometry->idToDet(recHits.front().geographicalId());
      //this is wrong because the FTS is defined at vertex, and it need to be properly propagated to the first rechit
      //      const TrajectoryStateOnSurface initialTSOS(initialFTS, initialLayer->surface());      
       const TrajectoryStateOnSurface initialTSOS = thePropagator->propagate(initialFTS,initialLayer->surface()) ;
       if (!initialTSOS.isValid()) continue; 
       

       PTSOD = trajectoryStateTransform::persistentState(initialTSOS,recHits.front().geographicalId().rawId()); 
    }
    
    TrackCandidate  
      newTrackCandidate(recHits, 
			*aSeed, 
			PTSOD, 
			edm::RefToBase<TrajectorySeed>(theSeeds,seednr));

    LogDebug("FastTracking")<< "\tSeed Information " << std::endl
				      << "\tSeed Direction = " << aSeed->direction() << std::endl
				      << "\tSeed StartingDet = " << aSeed->startingState().detId() << std::endl
				      << "\tTrajectory Parameters "	      << std::endl
				      << "\t\t detId  = "	      << newTrackCandidate.trajectoryStateOnDet().detId() 	      << std::endl
				      << "\t\t loc.px = "
				      << newTrackCandidate.trajectoryStateOnDet().parameters().momentum().x()    
				      << std::endl
				      << "\t\t loc.py = "
				      << newTrackCandidate.trajectoryStateOnDet().parameters().momentum().y()    
				      << std::endl
				      << "\t\t loc.pz = "
				      << newTrackCandidate.trajectoryStateOnDet().parameters().momentum().z()    
				      << std::endl
				      << "\t\t error  = ";

    bool newTrackCandidateIsDuplicate = isDuplicateCandidate(*output,newTrackCandidate);
    if (!newTrackCandidateIsDuplicate) output->push_back(newTrackCandidate);
    LogDebug("FastTracking")<<"filling a track candidate into the collection, now having: "<<output->size();
    
    }//loop over possible simtrack associated.
  }//loop over all possible seeds.
  
  // Save the track candidates in the event
  LogDebug("FastTracking") << "Saving " 
				     << output->size() << " track candidates and " 
				     << recoTracks->size() << " reco::Tracks ";
  // Save the track candidates
  e.put(output);



  // Save the tracking recHits

  edm::OrphanHandle <TrackingRecHitCollection> theRecoHits = e.put(recoHits );

  // Create the track extras and add the references to the rechits
  unsigned hits=0;
  unsigned nTracks = recoTracks->size();
  recoTrackExtras->reserve(nTracks); // To save some time at push_back
  for ( unsigned index = 0; index < nTracks; ++index ) { 
    //reco::TrackExtra aTrackExtra;
    reco::Track& aTrack = recoTracks->at(index);
    reco::TrackExtra aTrackExtra(aTrack.outerPosition(),
				 aTrack.outerMomentum(),
				 aTrack.outerOk(),
				 aTrack.innerPosition(),
				 aTrack.innerMomentum(),
				 aTrack.innerOk(),
				 aTrack.outerStateCovariance(),
				 aTrack.outerDetId(),
				 aTrack.innerStateCovariance(),
				 aTrack.innerDetId(),
				 aTrack.seedDirection(),
				 aTrack.seedRef());

    unsigned nHits = aTrack.recHitsSize();
    for ( unsigned int ih=0; ih<nHits; ++ih) {
      aTrackExtra.add(TrackingRecHitRef(theRecoHits,hits++));
    }
    recoTrackExtras->push_back(aTrackExtra);
  }
  

  // Save the track extras
  edm::OrphanHandle<reco::TrackExtraCollection> theRecoTrackExtras = e.put(recoTrackExtras);

  // Add the reference to the track extra in the tracks
  for ( unsigned index = 0; index<nTracks; ++index ) { 
    const reco::TrackExtraRef theTrackExtraRef(theRecoTrackExtras,index);
    (recoTracks->at(index)).setExtra(theTrackExtraRef);
  }

  // Save the tracks
  edm::OrphanHandle<reco::TrackCollection> theRecoTracks = e.put(recoTracks);

  // Save the trajectories
  edm::OrphanHandle<std::vector<Trajectory> > theRecoTrajectories = e.put(recoTrajectories);
  
  // Create and set the trajectory/track association map 
  for ( unsigned index = 0; index<nTracks; ++index ) { 
    edm::Ref<std::vector<Trajectory> > trajRef( theRecoTrajectories, index );
    edm::Ref<reco::TrackCollection>    tkRef( theRecoTracks, index );
    recoTrajTrackMap->insert(trajRef,tkRef);
  }


  // Save the association map.
  e.put(recoTrajTrackMap);

}

int 
TrackCandidateProducer::findId(const reco::Track& aTrack) const {
  int trackId = -1;
  trackingRecHit_iterator aHit = aTrack.recHitsBegin();
  trackingRecHit_iterator lastHit = aTrack.recHitsEnd();
  for ( ; aHit!=lastHit; ++aHit ) {
    if ( !aHit->get()->isValid() ) continue;
    //    const SiTrackerGSRecHit2D * rechit = (const SiTrackerGSRecHit2D*) (aHit->get());
    const SiTrackerGSMatchedRecHit2D * rechit = (const SiTrackerGSMatchedRecHit2D*) (aHit->get());
    trackId = rechit->simtrackId();
    break;
  }
  return trackId;
}

void 
TrackCandidateProducer::addSplitHits(const TrackerRecHit& theCurrentRecHit,
				     std::vector<TrackerRecHit>& theTrackerRecHits) { 
  
  const SiTrackerGSRecHit2D* mHit = theCurrentRecHit.matchedHit()->monoHit();
  const SiTrackerGSRecHit2D* sHit = theCurrentRecHit.matchedHit()->stereoHit();
  
  // Add the new hits
  if( mHit->simhitId() < sHit->simhitId() ) {
    
    theTrackerRecHits.push_back(TrackerRecHit(mHit,theCurrentRecHit));
    theTrackerRecHits.push_back(TrackerRecHit(sHit,theCurrentRecHit));
    
  } else {
    
    theTrackerRecHits.push_back(TrackerRecHit(sHit,theCurrentRecHit));
    theTrackerRecHits.push_back(TrackerRecHit(mHit,theCurrentRecHit));
    
  }

}

bool TrackCandidateProducer::isDuplicateCandidate(const TrackCandidateCollection& candidates, const TrackCandidate& newCand) const{
  typedef TrackCandidateCollection::const_iterator TCCI;
  
  TCCI candsEnd = candidates.end();
  double newQbp = newCand.trajectoryStateOnDet().parameters().qbp();
  double newDxdz = newCand.trajectoryStateOnDet().parameters().dxdz();
  TrackCandidate::range newHits = newCand.recHits();

  for (TCCI iCand = candidates.begin(); iCand!= candsEnd; ++iCand){
    //pick some first traits of duplication: qbp and dxdz
    double iQbp = iCand->trajectoryStateOnDet().parameters().qbp();
    double iDxdz = iCand->trajectoryStateOnDet().parameters().dxdz();
    if (newQbp == iQbp && newDxdz == iDxdz){
      LogDebug("isDuplicateCandidate")<<"Probably a duplicate "<<iQbp <<" : "<<iDxdz;
      TrackCandidate::range iHits = iCand->recHits();
      unsigned int nHits = 0;
      unsigned int nShared = 0;
      unsigned int nnewHits = 0;
      for (TrackCandidate::const_iterator iiHit = iHits.first; iiHit != iHits.second; ++iiHit){
	nHits++;
	for (TrackCandidate::const_iterator inewHit = newHits.first; inewHit != newHits.second; ++inewHit){
	  if (iiHit == iHits.first) nnewHits++;
	  if (sameLocalParameters(&*iiHit,&*inewHit)) nShared++;
	}
      }
      LogDebug("isDuplicateCandidate") <<"nHits  "<<nHits<<" nnewHits "<< nnewHits<< " nShared "<<nShared;
      if (nHits == nShared && nHits == nnewHits) return true;
    }
  }
  return false;
}

bool TrackCandidateProducer::sameLocalParameters(const TrackingRecHit* aH, const TrackingRecHit* bH) const {
  bool localPointSame = aH->localPosition() == bH->localPosition();
  bool localErrorSame = aH->localPositionError().xx() == bH->localPositionError().xx();
  localErrorSame &= aH->localPositionError().yy() == bH->localPositionError().yy();
  localErrorSame &= aH->localPositionError().xy() == bH->localPositionError().xy();
  return localPointSame && localErrorSame;
}
