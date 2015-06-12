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
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "FastSimulation/Tracking/interface/TrajectorySeedHitCandidate.h"
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

TrackCandidateProducer::TrackCandidateProducer(const edm::ParameterSet& conf)
{  
#ifdef FAMOS_DEBUG
  std::cout << "TrackCandidateProducer created" << std::endl;
#endif
    
  if(conf.exists("keepFittedTracks")){
    std::cout << "TrackCandidateProducer no longer supports option keepFittedTracks" << std::endl;
    assert(false);
  }
  if(conf.exists("TrackProducers")){
    edm::LogError("TrackCandidateProducer") << "TrackCandidateProducer no longer supports TrackProducers" << std::endl;
    exit (0);
  }
 
  // The main product is a track candidate collection.
  produces<TrackCandidateCollection>();
  
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

  estimatorCut_= conf.getParameter<double>("EstimatorCut");
  
  // input tags, labels, tokens
  edm::InputTag simTrackLabel = conf.getParameter<edm::InputTag>("simTracks");
  simVertexToken = consumes<edm::SimVertexContainer>(simTrackLabel);
  simTrackToken = consumes<edm::SimTrackContainer>(simTrackLabel);

  edm::InputTag seedLabel = conf.getParameter<edm::InputTag>("src");
  seedToken = consumes<edm::View<TrajectorySeed> >(seedLabel);

  edm::InputTag recHitLabel = conf.getParameter<edm::InputTag>("recHits");
  recHitToken = consumes<SiTrackerGSMatchedRecHit2DCollection>(recHitLabel);
  
  propagatorLabel = conf.getParameter<std::string>("propagator");
}
  
void 
TrackCandidateProducer::produce(edm::Event& e, const edm::EventSetup& es) {        

  // get services
  edm::ESHandle<MagneticField>          magneticField;
  es.get<IdealMagneticFieldRecord>().get(magneticField);

  edm::ESHandle<TrackerGeometry>        trackerGeometry;
  es.get<TrackerDigiGeometryRecord>().get(trackerGeometry);

  edm::ESHandle<TrackerTopology>        trackerTopology;
  es.get<TrackerTopologyRcd>().get(trackerTopology);

  edm::ESHandle<Propagator>             propagator;
  es.get<TrackingComponentsRecord>().get(propagatorLabel,propagator);
    
  // get products
  edm::Handle<edm::View<TrajectorySeed> > seeds;
  e.getByToken(seedToken,seeds);

  edm::Handle<SiTrackerGSMatchedRecHit2DCollection> recHits;
  e.getByToken(recHitToken, recHits);

  edm::Handle<edm::SimVertexContainer> simVertices;
  e.getByToken(simVertexToken,simVertices);

  edm::Handle<edm::SimTrackContainer> simTracks;
  e.getByToken(simTrackToken,simTracks);
  
  // Useful typedef's to avoid retyping
  typedef std::pair<reco::TrackRef,edm::Ref<std::vector<Trajectory> > > TrackPair;

  // The produced objects
  std::auto_ptr<TrackCandidateCollection> output(new TrackCandidateCollection);    

  //get other general things
  const std::vector<unsigned> simTrackIds = recHits->ids();
      
  // Loop over the seeds
  int currentTrackId = -1;
 
  unsigned seed_size = seeds->size(); 
  for (unsigned seednr = 0; seednr < seed_size; ++seednr){
    
    LogDebug("FastTracking")<<"looking at seed #:"<<seednr;

    // The seed
    const BasicTrajectorySeed* aSeed = &((*seeds)[seednr]);

    std::vector<int> simTrackIds;
    std::map<int,TrajectoryStateOnSurface> seedStates;
    std::map<int,TrajectoryStateOnSurface> simtkStates;

    TrajectorySeedHitCandidate theFirstSeedingTrackerRecHit;
    if (seeds->at(seednr).nHits()==0){
      //new stuff for no hits on seed

      LogDebug("FastTracking")<<" seed with no hits to be considered.";

      PTrajectoryStateOnDet ptod =seeds->at(seednr).startingState();
      DetId id(ptod.detId());
      const GeomDet * g = trackerGeometry->idToDet(id);
      const Surface * surface=&g->surface();
      
      TrajectoryStateOnSurface seedState(trajectoryStateTransform::transientState(ptod,surface,magneticField.product()));
      
      edm::ESHandle<Propagator> _propagator;
      es.get<TrackingComponentsRecord>().get("AnyDirectionAnalyticalPropagator",_propagator);
      
      double minimunEst=1000000;
      LogDebug("FastTracking")<<"looking at: "<< simTrackIds.size()<<" simtracks.";
      for ( unsigned tkId=0;  tkId != simTrackIds.size(); ++tkId ) {
	
	const SimTrack & simtrack =simTracks->at(simTrackIds[tkId]);

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
						  magneticField.product());
	FreeTrajectoryState simtrack_trackerstate(glb_parameters);
	
	TrajectoryStateOnSurface simtrack_comparestate = _propagator->propagate(simtrack_trackerstate,*surface);

	  
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
	  simTrackIds.push_back(simTrackIds[tkId]);
	  //making a state with exactly the sim track parameters
	  //the initial errors are set to unity just for kicks
	  //	  AlgebraicSymMatrix C(5,1);// C*=50;
	  //new attempt!!!!
	  AlgebraicSymMatrix55 C = seedState.curvilinearError().matrix();
	  C *= 0.0000001;

	  seedStates[simTrackIds[tkId]] = TrajectoryStateOnSurface(simtrack_comparestate.globalParameters(),
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
      theFirstSeedingTrackerRecHit = TrajectorySeedHitCandidate(theFirstSeedingRecHit,trackerGeometry.product(),trackerTopology.product());
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
      std::vector<TrajectorySeedHitCandidate> theTrackerRecHits;
      unsigned theNumberOfCrossedLayers = 0;      

	LogDebug("FastTracking")<<"Track " << simTrackId << " is considered to return a track candidate" ;

	// Get all the rechits associated to this track
	SiTrackerGSMatchedRecHit2DCollection::range theRecHitRange = recHits->get(simTrackId);
	SiTrackerGSMatchedRecHit2DCollection::const_iterator theRecHitRangeIteratorBegin = theRecHitRange.first;
	SiTrackerGSMatchedRecHit2DCollection::const_iterator theRecHitRangeIteratorEnd   = theRecHitRange.second;
	SiTrackerGSMatchedRecHit2DCollection::const_iterator iterRecHit;

	LogDebug("FastTracking")<<"counting: "<<theRecHitRangeIteratorEnd-theRecHitRangeIteratorBegin<<" hits to be considered.";

	bool firstRecHit = true;
	TrajectorySeedHitCandidate theCurrentRecHit, thePreviousRecHit;
	TrajectorySeedHitCandidate theFirstHitComp, theSecondHitComp;
	
	for ( iterRecHit = theRecHitRangeIteratorBegin; 
	      iterRecHit != theRecHitRangeIteratorEnd; 
	      ++iterRecHit) {
	  
	  // Check the number of crossed layers
	  if ( theNumberOfCrossedLayers >= maxNumberOfCrossedLayers ) continue;
	  
	  // Get current and previous rechits
	  if(!firstRecHit) thePreviousRecHit = theCurrentRecHit;
	  theCurrentRecHit = TrajectorySeedHitCandidate(&(*iterRecHit),trackerGeometry.product(),trackerTopology.product());
	  
	  //>>>>>>>>>BACKBUILDING CHANGE: DO NOT STAT FROM THE FIRST HIT OF THE SEED

	  // NOTE: checking the direction --> specific for OIHit only
	  //	  if( aSeed->direction()!=oppositeToMomentum ) { 
	  //  // Check that the first rechit is indeed the first seeding hit
	  //  if ( firstRecHit && theCurrentRecHit != theFirstSeedingTrackerRecHit && seeds->at(seednr).nHits()!=0 ) continue;
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
	}// End of loop over the track rechits

    
      LogDebug("FastTracking")<<" number of hits: " << theTrackerRecHits.size()<<" after counting overlaps and splitting.";

      // 1) Create the OwnVector of TrackingRecHits
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
      int vertexIndex = (*simTracks)[currentTrackId].vertIndex();
      //   a) origin vertex
      GlobalPoint  position(simVertices->at(vertexIndex).position().x(),
			    simVertices->at(vertexIndex).position().y(),
			    simVertices->at(vertexIndex).position().z());
      
      //   b) initial momentum
      GlobalVector momentum( simTracks->at(currentTrackId).momentum().x() , 
			     simTracks->at(currentTrackId).momentum().y() , 
			     simTracks->at(currentTrackId).momentum().z() );
      //   c) electric charge
      float        charge   = (*simTracks)[simTrackId].charge();
      //  -> inital parameters
      GlobalTrajectoryParameters initialParams(position,momentum,(int)charge,magneticField.product());
 //  -> large initial errors
      AlgebraicSymMatrix55 errorMatrix= AlgebraicMatrixID();    
      CurvilinearTrajectoryError initialError(errorMatrix);
      // -> initial state
      FreeTrajectoryState initialFTS(initialParams, initialError);      
#ifdef FAMOS_DEBUG
      std::cout << "TrajectorySeedProducer: FTS momentum " << initialFTS.momentum() << std::endl;
#endif
      const GeomDet* initialLayer = trackerGeometry->idToDet(recHits.front().geographicalId());
      //this is wrong because the FTS is defined at vertex, and it need to be properly propagated to the first rechit
      //      const TrajectoryStateOnSurface initialTSOS(initialFTS, initialLayer->surface());      
       const TrajectoryStateOnSurface initialTSOS = propagator->propagate(initialFTS,initialLayer->surface()) ;
       if (!initialTSOS.isValid()) continue; 
       

       PTSOD = trajectoryStateTransform::persistentState(initialTSOS,recHits.front().geographicalId().rawId()); 
    }
    
    TrackCandidate newTrackCandidate(recHits,*aSeed,PTSOD,edm::RefToBase<TrajectorySeed>(seeds,seednr));

    output->push_back(newTrackCandidate);
    
    }//loop over possible simtrack associated.
  }//loop over all possible seeds.
  
  // Save the track candidates in the event
  LogDebug("FastTracking") << "Saving " 
			   << output->size() << " track candidates" ;
  // Save the track candidates
  e.put(output);
}

void 
TrackCandidateProducer::addSplitHits(const TrajectorySeedHitCandidate& theCurrentRecHit,
				     std::vector<TrajectorySeedHitCandidate>& theTrackerRecHits) { 
  
  const SiTrackerGSRecHit2D* mHit = theCurrentRecHit.matchedHit()->monoHit();
  const SiTrackerGSRecHit2D* sHit = theCurrentRecHit.matchedHit()->stereoHit();
  
  // Add the new hits
  if( mHit->simhitId() < sHit->simhitId() ) {
    
    theTrackerRecHits.push_back(TrajectorySeedHitCandidate(mHit,theCurrentRecHit));
    theTrackerRecHits.push_back(TrajectorySeedHitCandidate(sHit,theCurrentRecHit));
    
  } else {
    
    theTrackerRecHits.push_back(TrajectorySeedHitCandidate(sHit,theCurrentRecHit));
    theTrackerRecHits.push_back(TrajectorySeedHitCandidate(mHit,theCurrentRecHit));
    
  }
}

