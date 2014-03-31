#include <memory>
#include <string>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include <FWCore/Utilities/interface/ESInputTag.h>

#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/Common/interface/View.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "RecoTracker/CkfPattern/interface/CkfTrackCandidateMakerBase.h"
#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"


#include "RecoTracker/CkfPattern/interface/SeedCleanerByHitPosition.h"
#include "RecoTracker/CkfPattern/interface/CachingSeedCleanerByHitPosition.h"
#include "RecoTracker/CkfPattern/interface/SeedCleanerBySharedInput.h"
#include "RecoTracker/CkfPattern/interface/CachingSeedCleanerBySharedInput.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"

#include "RecoTracker/TransientTrackingRecHit/interface/Traj2TrackHits.h"

#include<algorithm>
#include<functional>

#include "RecoTracker/CkfPattern/interface/PrintoutHelper.h"

using namespace edm;
using namespace std;

namespace cms{
  CkfTrackCandidateMakerBase::CkfTrackCandidateMakerBase(edm::ParameterSet const& conf, edm::ConsumesCollector && iC) : 

    conf_(conf),
    theTrackCandidateOutput(true),
    theTrajectoryOutput(false),
    useSplitting(conf.getParameter<bool>("useHitsSplitting")),
    doSeedingRegionRebuilding(conf.getParameter<bool>("doSeedingRegionRebuilding")),
    cleanTrajectoryAfterInOut(conf.getParameter<bool>("cleanTrajectoryAfterInOut")),
    reverseTrajectories(conf.existsAs<bool>("reverseTrajectories") && conf.getParameter<bool>("reverseTrajectories")),
    theMaxNSeeds(conf.getParameter<unsigned int>("maxNSeeds")),
    theTrajectoryBuilderName(conf.getParameter<std::string>("TrajectoryBuilder")), 
    theTrajectoryBuilder(0),
    theTrajectoryCleanerName(conf.getParameter<std::string>("TrajectoryCleaner")), 
    theTrajectoryCleaner(0),
    theInitialState(0),
    theNavigationSchoolName(conf.getParameter<std::string>("NavigationSchool")),
    theNavigationSchool(0),
    theSeedCleaner(0),
    maxSeedsBeforeCleaning_(0),
    theMTELabel(iC.consumes<MeasurementTrackerEvent>(conf.getParameter<edm::InputTag>("MeasurementTrackerEvent"))),
    skipClusters_(false)
  {  
    //produces<TrackCandidateCollection>();  
    // old configuration totally descoped.
    //    if (!conf.exists("src"))
    //      theSeedLabel = InputTag(conf_.getParameter<std::string>("SeedProducer"),conf_.getParameter<std::string>("SeedLabel"));
    //    else
      theSeedLabel= iC.consumes<edm::View<TrajectorySeed> >(conf.getParameter<edm::InputTag>("src"));
      if ( conf.exists("maxSeedsBeforeCleaning") ) 
	   maxSeedsBeforeCleaning_=conf.getParameter<unsigned int>("maxSeedsBeforeCleaning");

      if (conf.existsAs<edm::InputTag>("clustersToSkip")) {
        skipClusters_ = true;
        maskPixels_ = iC.consumes<PixelClusterMask>(conf.getParameter<edm::InputTag>("clustersToSkip"));
        maskStrips_ = iC.consumes<StripClusterMask>(conf.getParameter<edm::InputTag>("clustersToSkip"));
      }

    std::string cleaner = conf_.getParameter<std::string>("RedundantSeedCleaner");
    if (cleaner == "SeedCleanerByHitPosition") {
        theSeedCleaner = new SeedCleanerByHitPosition();
    } else if (cleaner == "SeedCleanerBySharedInput") {
        theSeedCleaner = new SeedCleanerBySharedInput();
    } else if (cleaner == "CachingSeedCleanerByHitPosition") {
        theSeedCleaner = new CachingSeedCleanerByHitPosition();
    } else if (cleaner == "CachingSeedCleanerBySharedInput") {
      int numHitsForSeedCleaner = conf_.existsAs<int>("numHitsForSeedCleaner") ? 
	conf_.getParameter<int>("numHitsForSeedCleaner") : 4;
      int onlyPixelHits = conf_.existsAs<bool>("onlyPixelHitsForSeedCleaner") ? 
	conf_.getParameter<bool>("onlyPixelHitsForSeedCleaner") : false;
      theSeedCleaner = new CachingSeedCleanerBySharedInput(numHitsForSeedCleaner,onlyPixelHits);
    } else if (cleaner == "none") {
        theSeedCleaner = 0;
    } else {
        throw cms::Exception("RedundantSeedCleaner not found", cleaner);
    }


  }

  
  // Virtual destructor needed.
  CkfTrackCandidateMakerBase::~CkfTrackCandidateMakerBase() {
    delete theInitialState;  
    if (theSeedCleaner) delete theSeedCleaner;
  }  

  void CkfTrackCandidateMakerBase::beginRunBase (edm::Run const & r, EventSetup const & es)
  {
    /* no op*/
  }

  void CkfTrackCandidateMakerBase::setEventSetup( const edm::EventSetup& es ) {

    //services
    es.get<TrackerRecoGeometryRecord>().get( theGeomSearchTracker );
    std::string mfName = "";
    if (conf_.exists("SimpleMagneticField"))
      mfName = conf_.getParameter<std::string>("SimpleMagneticField");
    es.get<IdealMagneticFieldRecord>().get(mfName,theMagField );
    //    edm::ESInputTag mfESInputTag(mfName);
    //    es.get<IdealMagneticFieldRecord>().get(mfESInputTag,theMagField );

    if (!theInitialState){
      // constructor uses the EventSetup, it must be in the setEventSetup were it has a proper value.
      // get nested parameter set for the TransientInitialStateEstimator
      ParameterSet tise_params = conf_.getParameter<ParameterSet>("TransientInitialStateEstimatorParameters") ;
      theInitialState          = new TransientInitialStateEstimator( es,tise_params);
    }

    theInitialState->setEventSetup( es );

    edm::ESHandle<TrajectoryCleaner> trajectoryCleanerH;
    es.get<TrajectoryCleaner::Record>().get(theTrajectoryCleanerName, trajectoryCleanerH);
    theTrajectoryCleaner= trajectoryCleanerH.product();

    edm::ESHandle<NavigationSchool> navigationSchoolH;
    es.get<NavigationSchoolRecord>().get(theNavigationSchoolName, navigationSchoolH);
    theNavigationSchool = navigationSchoolH.product();

    // set the TrajectoryBuilder
    edm::ESHandle<TrajectoryBuilder> theTrajectoryBuilderHandle;
    es.get<CkfComponentsRecord>().get(theTrajectoryBuilderName,theTrajectoryBuilderHandle);
    theTrajectoryBuilder = dynamic_cast<const BaseCkfTrajectoryBuilder*>(theTrajectoryBuilderHandle.product());    
    assert(theTrajectoryBuilder);
  }

  // Functions that gets called by framework every event
  void CkfTrackCandidateMakerBase::produceBase(edm::Event& e, const edm::EventSetup& es)
  { 
    // getting objects from the EventSetup
    setEventSetup( es ); 

    // set the correct navigation
    NavigationSetter setter( *theNavigationSchool);
    
    // propagator
    edm::ESHandle<Propagator> thePropagator;
    es.get<TrackingComponentsRecord>().get("AnyDirectionAnalyticalPropagator",
					   thePropagator);

    // method for Debugging
    printHitsDebugger(e);

    // Step A: set Event for the TrajectoryBuilder
    edm::Handle<MeasurementTrackerEvent> data;
    e.getByToken(theMTELabel, data);

    std::auto_ptr<BaseCkfTrajectoryBuilder> trajectoryBuilder;
    std::auto_ptr<MeasurementTrackerEvent> dataWithMasks;
    if (skipClusters_) {
        edm::Handle<PixelClusterMask> pixelMask;
        e.getByToken(maskPixels_, pixelMask);
            edm::Handle<StripClusterMask> stripMask;
            e.getByToken(maskStrips_, stripMask);
            dataWithMasks.reset(new MeasurementTrackerEvent(*data, *stripMask, *pixelMask));
        //std::cout << "Trajectory builder " << conf_.getParameter<std::string>("@module_label") << " created with masks, " << std::endl;
        trajectoryBuilder.reset(theTrajectoryBuilder->clone(&*dataWithMasks));
    } else {
        //std::cout << "Trajectory builder " << conf_.getParameter<std::string>("@module_label") << " created without masks, " << std::endl;
        trajectoryBuilder.reset(theTrajectoryBuilder->clone(&*data));
    }
    
    // Step B: Retrieve seeds
    
    edm::Handle<View<TrajectorySeed> > collseed;
    e.getByToken(theSeedLabel, collseed);
    
    // Step C: Create empty output collection
    std::auto_ptr<TrackCandidateCollection> output(new TrackCandidateCollection);    
    std::auto_ptr<std::vector<Trajectory> > outputT (new std::vector<Trajectory>());

    if ( (*collseed).size()>theMaxNSeeds ) {
      LogError("TooManySeeds")<<"Exceeded maximum numeber of seeds! theMaxNSeeds="<<theMaxNSeeds<<" nSeed="<<(*collseed).size();
      if (theTrackCandidateOutput){ e.put(output);}
      if (theTrajectoryOutput){e.put(outputT);}
      return;
    }
    
    // Step D: Invoke the building algorithm
    if ((*collseed).size()>0){

      unsigned int lastCleanResult=0;
      vector<Trajectory> rawResult;
      rawResult.reserve(collseed->size() * 4);

      if (theSeedCleaner) theSeedCleaner->init( &rawResult );
      
      // method for debugging
      countSeedsDebugger();

      vector<Trajectory> theTmpTrajectories;

      // Loop over seeds
      size_t collseed_size = collseed->size(); 
      for (size_t j = 0; j < collseed_size; j++){

	LogDebug("CkfPattern") << "======== Begin to look for trajectories from seed " << j << " ========"<<endl;
	
	// Check if seed hits already used by another track
	if (theSeedCleaner && !theSeedCleaner->good( &((*collseed)[j])) ) {
          LogDebug("CkfTrackCandidateMakerBase")<<" Seed cleaning kills seed "<<j;
          continue; 
        }

	// Build trajectory from seed outwards
        theTmpTrajectories.clear();
	auto const & startTraj = trajectoryBuilder->buildTrajectories( (*collseed)[j], theTmpTrajectories, nullptr );
	
       
	LogDebug("CkfPattern") << "======== In-out trajectory building found " << theTmpTrajectories.size()
			            << " trajectories from seed " << j << " ========"<<endl
			       <<PrintoutHelper::dumpCandidates(theTmpTrajectories);
	
        if (cleanTrajectoryAfterInOut) {

	  // Select the best trajectory from this seed (declare others invalid)
  	  theTrajectoryCleaner->clean(theTmpTrajectories);

  	  LogDebug("CkfPattern") << "======== In-out trajectory cleaning gave the following valid trajectories from seed " 
                                 << j << " ========"<<endl
				 << PrintoutHelper::dumpCandidates(theTmpTrajectories);
        }

	// Optionally continue building trajectory back through 
	// seed and if possible further inwards.
	
	if (doSeedingRegionRebuilding) {
	  trajectoryBuilder->rebuildTrajectories(startTraj,(*collseed)[j],theTmpTrajectories);      

  	  LogDebug("CkfPattern") << "======== Out-in trajectory building found " << theTmpTrajectories.size()
  			              << " valid/invalid trajectories from seed " << j << " ========"<<endl
				 <<PrintoutHelper::dumpCandidates(theTmpTrajectories);
        }
	

        // Select the best trajectory from this seed (after seed region rebuilding, can be more than one)
	theTrajectoryCleaner->clean(theTmpTrajectories);

        LogDebug("CkfPattern") << "======== Trajectory cleaning gave the following valid trajectories from seed " 
                               << j << " ========"<<endl
			       <<PrintoutHelper::dumpCandidates(theTmpTrajectories);

	for(vector<Trajectory>::iterator it=theTmpTrajectories.begin();
	    it!=theTmpTrajectories.end(); it++){
	  if( it->isValid() ) {
	    it->setSeedRef(collseed->refAt(j));
	    // Store trajectory
	    rawResult.push_back(std::move(*it));
  	    // Tell seed cleaner which hits this trajectory used.
            //TO BE FIXED: this cut should be configurable via cfi file
            if (theSeedCleaner && rawResult.back().foundHits()>3) theSeedCleaner->add( &rawResult.back() );
            //if (theSeedCleaner ) theSeedCleaner->add( & (*it) );
	  }
	}

        theTmpTrajectories.clear();
        
	LogDebug("CkfPattern") << "rawResult trajectories found so far = " << rawResult.size();

	if ( maxSeedsBeforeCleaning_ >0 && rawResult.size() > maxSeedsBeforeCleaning_+lastCleanResult) {
          theTrajectoryCleaner->clean(rawResult);
          rawResult.erase(std::remove_if(rawResult.begin()+lastCleanResult,rawResult.end(),
					 std::not1(std::mem_fun_ref(&Trajectory::isValid))),
			  rawResult.end());
          lastCleanResult=rawResult.size();
        }

      }
      // end of loop over seeds
      
      if (theSeedCleaner) theSeedCleaner->done();
   
      // std::cout << "VICkfPattern " << "rawResult trajectories found = " << rawResult.size() << std::endl;

   
      // Step E: Clean the results to avoid duplicate tracks
      // Rejected ones just flagged as invalid.
      theTrajectoryCleaner->clean(rawResult);

      LogDebug("CkfPattern") << "======== Final cleaning of entire event found " << rawResult.size() 
                             << " valid/invalid trajectories ======="<<endl
			     <<PrintoutHelper::dumpCandidates(rawResult);

      LogDebug("CkfPattern") << "removing invalid trajectories.";

      vector<Trajectory> & unsmoothedResult(rawResult);
      unsmoothedResult.erase(std::remove_if(unsmoothedResult.begin(),unsmoothedResult.end(),
					    std::not1(std::mem_fun_ref(&Trajectory::isValid))),
			     unsmoothedResult.end());
      
      // If requested, reverse the trajectories creating a new 1-hit seed on the last measurement of the track
      if (reverseTrajectories) {
        for (auto it = unsmoothedResult.begin(), ed = unsmoothedResult.end(); it != ed; ++it) {
          // reverse the trajectory only if it has valid hit on the last measurement (should happen)
          if (it->lastMeasurement().updatedState().isValid() && 
              it->lastMeasurement().recHit().get() != 0     &&
              it->lastMeasurement().recHit()->isValid()) {
            // I can't use reverse in place, because I want to change the seed
            // 1) reverse propagation direction
            PropagationDirection direction = it->direction();
            if (direction == alongMomentum)           direction = oppositeToMomentum;
            else if (direction == oppositeToMomentum) direction = alongMomentum;
            // 2) make a seed
            TrajectoryStateOnSurface const & initState = it->lastMeasurement().updatedState();
            auto                    initId = it->lastMeasurement().recHitR().rawId();
            PTrajectoryStateOnDet && state = trajectoryStateTransform::persistentState( initState, initId);
            TrajectorySeed::recHitContainer hits; 
            hits.push_back(it->lastMeasurement().recHit()->hit()->clone());
            boost::shared_ptr<const TrajectorySeed> seed(new TrajectorySeed(state, std::move(hits), direction));
            // 3) make a trajectory
            Trajectory trajectory(seed, direction);
	    trajectory.setNLoops(it->nLoops());
            trajectory.setSeedRef(it->seedRef());
            // 4) push states in reversed order
            Trajectory::DataContainer &meas = it->measurements();
            for (auto itmeas = meas.rbegin(), endmeas = meas.rend(); itmeas != endmeas; ++itmeas) {
              trajectory.push(std::move(*itmeas));
            }
            // replace
            (*it)= std::move(trajectory); 
          } else {
            edm::LogWarning("CkfPattern_InvalidLastMeasurement") << "Last measurement of the trajectory is invalid, cannot reverse it";
          }     
        }
      }

   
      int viTotHits=0;   
   
      if (theTrackCandidateOutput){
	// Step F: Convert to TrackCandidates
       output->reserve(unsmoothedResult.size());
       Traj2TrackHits t2t(theTrajectoryBuilder->hitBuilder(),true);

       for (vector<Trajectory>::const_iterator it = unsmoothedResult.begin();
	    it != unsmoothedResult.end(); ++it) {
	
	 LogDebug("CkfPattern") << "copying "<<(useSplitting?"splitted":"un-splitted")<<" hits from trajectory";
	 edm::OwnVector<TrackingRecHit> recHits;
         if(it->direction() != alongMomentum) LogDebug("CkfPattern") << "not along momentum... " << std::endl;
         t2t(*it,recHits,useSplitting);

         viTotHits+=recHits.size();        

	 LogDebug("CkfPattern") << "getting initial state.";
	 const bool doBackFit = (!doSeedingRegionRebuilding) & (!reverseTrajectories);
	 std::pair<TrajectoryStateOnSurface, const GeomDet*> && initState = theInitialState->innerState( *it , doBackFit);

	 // temporary protection againt invalid initial states
	 if ( !initState.first.isValid() || initState.second == nullptr || edm::isNotFinite(initState.first.globalPosition().x())) {
	   //cout << "invalid innerState, will not make TrackCandidate" << endl;
	   continue;
	 }
	 
	 PTrajectoryStateOnDet state;
	 if(useSplitting && (initState.second != recHits.front().det()) && recHits.front().det() ){	 
	   LogDebug("CkfPattern") << "propagating to hit front in case of splitting.";
	   TrajectoryStateOnSurface && propagated = thePropagator->propagate(initState.first,recHits.front().det()->surface());
	   if (!propagated.isValid()) continue;
	   state = trajectoryStateTransform::persistentState(propagated,
								      recHits.front().rawId());
	 }
	 else state = trajectoryStateTransform::persistentState( initState.first,
							         initState.second->geographicalId().rawId());
	 LogDebug("CkfPattern") << "pushing a TrackCandidate.";
	 output->emplace_back(recHits,it->seed(),state,it->seedRef(),it->nLoops());
       }
      }//output trackcandidates

      edm::ESHandle<TrackerGeometry> tracker;
      es.get<TrackerDigiGeometryRecord>().get(tracker);            
      LogTrace("CkfPattern|TrackingRegressionTest") << "========== CkfTrackCandidateMaker Info =========="
						    << "number of Seed: " << collseed->size()<<endl
      						    <<PrintoutHelper::regressionTest(*tracker,unsmoothedResult);

      assert(viTotHits>=0); // just to use it...
      // std::cout << "VICkfPattern result " << output->size() << " " << viTotHits << std::endl;
     
      if (theTrajectoryOutput){ outputT->swap(unsmoothedResult);}

    }// end of ((*collseed).size()>0)
    
    // method for debugging
    deleteAssocDebugger();

    // Step G: write output to file
    if (theTrackCandidateOutput){ e.put(output);}
    if (theTrajectoryOutput){e.put(outputT);}
  }
  
}

