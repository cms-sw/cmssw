#include <memory>
#include <string>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"

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

#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"

#include<algorithm>
#include<functional>

#include "RecoTracker/CkfPattern/interface/PrintoutHelper.h"

using namespace edm;
using namespace std;

namespace cms{
  CkfTrackCandidateMakerBase::CkfTrackCandidateMakerBase(edm::ParameterSet const& conf) : 

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
    maxSeedsBeforeCleaning_(0)
  {  
    //produces<TrackCandidateCollection>();  
    // old configuration totally descoped.
    //    if (!conf.exists("src"))
    //      theSeedLabel = InputTag(conf_.getParameter<std::string>("SeedProducer"),conf_.getParameter<std::string>("SeedLabel"));
    //    else
      theSeedLabel= conf.getParameter<edm::InputTag>("src");
      if ( conf.exists("maxSeedsBeforeCleaning") ) 
	   maxSeedsBeforeCleaning_=conf.getParameter<unsigned int>("maxSeedsBeforeCleaning");

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

  void CkfTrackCandidateMakerBase::beginRunBase (edm::Run & r, EventSetup const & es)
  {
    /* no op*/
  }

  void CkfTrackCandidateMakerBase::setEventSetup( const edm::EventSetup& es ) {

    //services
    es.get<TrackerRecoGeometryRecord>().get( theGeomSearchTracker );
    es.get<IdealMagneticFieldRecord>().get( theMagField );

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
    theTrajectoryBuilder->setEvent(e);        
    
    // Step B: Retrieve seeds
    
    edm::Handle<View<TrajectorySeed> > collseed;
    e.getByLabel(theSeedLabel, collseed);
    
    // Step C: Create empty output collection
    std::auto_ptr<TrackCandidateCollection> output(new TrackCandidateCollection);    
    std::auto_ptr<std::vector<Trajectory> > outputT (new std::vector<Trajectory>());

    if ( (*collseed).size()>theMaxNSeeds ) {
      LogError("TooManySeeds")<<"Exceeded maximum numeber of seeds! theMaxNSeeds="<<theMaxNSeeds<<" nSeed="<<(*collseed).size();
      if (theTrackCandidateOutput){ e.put(output);}
      if (theTrajectoryOutput){e.put(outputT);}
      theTrajectoryBuilder->unset();
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
       
	// Check if seed hits already used by another track
	if (theSeedCleaner && !theSeedCleaner->good( &((*collseed)[j])) ) {
          LogDebug("CkfTrackCandidateMakerBase")<<" Seed cleaning kills seed "<<j;
          continue; 
        }

	// Build trajectory from seed outwards
        theTmpTrajectories.clear();
	auto const & startTraj = theTrajectoryBuilder->buildTrajectories( (*collseed)[j], theTmpTrajectories, nullptr );
	
       
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
	  theTrajectoryBuilder->rebuildTrajectories(startTraj,(*collseed)[j],theTmpTrajectories);      

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
	    rawResult.push_back(*it);
  	    // Tell seed cleaner which hits this trajectory used.
            //TO BE FIXED: this cut should be configurable via cfi file
            if (theSeedCleaner && it->foundHits()>3) theSeedCleaner->add( & (*it) );
            //if (theSeedCleaner ) theSeedCleaner->add( & (*it) );
	  }
	}

        theTmpTrajectories.clear();
        
	LogDebug("CkfPattern") << "rawResult trajectories found so far = " << rawResult.size();

	if ( maxSeedsBeforeCleaning_ >0 && rawResult.size() > maxSeedsBeforeCleaning_+lastCleanResult) {
          theTrajectoryCleaner->clean(rawResult);
          rawResult.erase(std::remove_if(rawResult.begin(),rawResult.end(),
					 std::not1(std::mem_fun_ref(&Trajectory::isValid))),
			  rawResult.end());
          lastCleanResult=rawResult.size();
        }

      }
      // end of loop over seeds
      
      if (theSeedCleaner) theSeedCleaner->done();
      
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
        vector<Trajectory> reversed; 
        reversed.reserve(unsmoothedResult.size());
        for (vector<Trajectory>::const_iterator it = unsmoothedResult.begin(), ed = unsmoothedResult.end(); it != ed; ++it) {
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
            TrajectoryStateOnSurface initState = it->lastMeasurement().updatedState();
            DetId                    initDetId = it->lastMeasurement().recHit()->geographicalId();
            PTrajectoryStateOnDet state = trajectoryStateTransform::persistentState( initState, initDetId.rawId());
            TrajectorySeed::recHitContainer hits; 
            hits.push_back(*it->lastMeasurement().recHit()->hit());
            boost::shared_ptr<const TrajectorySeed> seed(new TrajectorySeed(state, hits, direction));
            // 3) make a trajectory
            Trajectory trajectory(seed, direction);
	    trajectory.setNLoops(it->nLoops());
            trajectory.setSeedRef(it->seedRef());
            // 4) push states in reversed order
            const Trajectory::DataContainer &meas = it->measurements();
            for (Trajectory::DataContainer::const_reverse_iterator itmeas = meas.rbegin(), endmeas = meas.rend(); itmeas != endmeas; ++itmeas) {
              trajectory.push(*itmeas);
            } 
            reversed.push_back(trajectory);
          } else {
            edm::LogWarning("CkfPattern_InvalidLastMeasurement") << "Last measurement of the trajectory is invalid, cannot reverse it";
            reversed.push_back(*it);
          }     
        }
        unsmoothedResult.swap(reversed);
      }

      //      for (vector<Trajectory>::const_iterator itraw = rawResult.begin();
      //	   itraw != rawResult.end(); itraw++) {
      //if((*itraw).isValid()) unsmoothedResult.push_back( *itraw);
      //}

      //analyseCleanedTrajectories(unsmoothedResult);
      
      if (theTrackCandidateOutput){
	// Step F: Convert to TrackCandidates
       output->reserve(unsmoothedResult.size());
       for (vector<Trajectory>::const_iterator it = unsmoothedResult.begin();
	    it != unsmoothedResult.end(); ++it) {
	
	 Trajectory::RecHitContainer thits;
	 //it->recHitsV(thits);
	 LogDebug("CkfPattern") << "retrieving "<<(useSplitting?"splitted":"un-splitted")<<" hits from trajectory";
	 it->recHitsV(thits,useSplitting);
	 OwnVector<TrackingRecHit> recHits;
	 recHits.reserve(thits.size());
	 LogDebug("CkfPattern") << "cloning hits into new collection.";
	 for (Trajectory::RecHitContainer::const_iterator hitIt = thits.begin();
	      hitIt != thits.end(); ++hitIt) {
	   recHits.push_back( (**hitIt).hit()->clone());
	 }

	 LogDebug("CkfPattern") << "getting initial state.";
	 const bool doBackFit = !doSeedingRegionRebuilding && !reverseTrajectories;
	 std::pair<TrajectoryStateOnSurface, const GeomDet*> initState = 
	   theInitialState->innerState( *it , doBackFit);

	 // temporary protection againt invalid initial states
	 if (! initState.first.isValid() || initState.second == 0 || edm::isNotFinite(initState.first.globalPosition().x())) {
	   //cout << "invalid innerState, will not make TrackCandidate" << endl;
	   continue;
	 }
	 
	 PTrajectoryStateOnDet state;
	 if(useSplitting && (initState.second != thits.front()->det()) && thits.front()->det() ){	 
	   LogDebug("CkfPattern") << "propagating to hit front in case of splitting.";
	   TrajectoryStateOnSurface propagated = thePropagator->propagate(initState.first,thits.front()->det()->surface());
	   if (!propagated.isValid()) continue;
	   state = trajectoryStateTransform::persistentState(propagated,
								      thits.front()->det()->geographicalId().rawId());
	 }
	 else state = trajectoryStateTransform::persistentState( initState.first,
									initState.second->geographicalId().rawId());
	 LogDebug("CkfPattern") << "pushing a TrackCandidate.";
	 output->push_back(TrackCandidate(recHits,it->seed(),state,it->seedRef(),it->nLoops() ) );
       }
      }//output trackcandidates

      edm::ESHandle<TrackerGeometry> tracker;
      es.get<TrackerDigiGeometryRecord>().get(tracker);            
      LogTrace("CkfPattern|TrackingRegressionTest") << "========== CkfTrackCandidateMaker Info =========="
						    << "number of Seed: " << collseed->size()<<endl
      						    <<PrintoutHelper::regressionTest(*tracker,unsmoothedResult);

      
     
      if (theTrajectoryOutput){ outputT->swap(unsmoothedResult);}

    }// end of ((*collseed).size()>0)
    
    // method for debugging
    deleteAssocDebugger();

    // Step G: write output to file
    if (theTrackCandidateOutput){ e.put(output);}
    if (theTrajectoryOutput){e.put(outputT);}
    
    //reset the MT.
    theTrajectoryBuilder->unset();
  }
  
}

