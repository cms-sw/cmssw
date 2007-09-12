#include <memory>
#include <string>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/OwnVector.h"
//view #include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
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

#include "RecoTracker/TkNavigation/interface/NavigationSchoolFactory.h"

#include<algorithm>
#include<functional>

using namespace edm;
using namespace std;

namespace cms{
  CkfTrackCandidateMakerBase::CkfTrackCandidateMakerBase(edm::ParameterSet const& conf) : 

    conf_(conf),theTrajectoryBuilder(0),theTrajectoryCleaner(0),
    theInitialState(0),theNavigationSchool(0),theSeedCleaner(0)
  {  
    //produces<TrackCandidateCollection>();  
  }

  
  // Virtual destructor needed.
  CkfTrackCandidateMakerBase::~CkfTrackCandidateMakerBase() {
    delete theInitialState;  
    delete theNavigationSchool;
    delete theTrajectoryCleaner;    
    if (theSeedCleaner) delete theSeedCleaner;
  }  

  void CkfTrackCandidateMakerBase::beginJobBase (EventSetup const & es)
  {
    //services
    es.get<TrackerRecoGeometryRecord>().get( theGeomSearchTracker );
    es.get<IdealMagneticFieldRecord>().get(theMagField);
      
    // get nested parameter set for the TransientInitialStateEstimator
    ParameterSet tise_params = conf_.getParameter<ParameterSet>("TransientInitialStateEstimatorParameters") ;
    theInitialState          = new TransientInitialStateEstimator( es,tise_params);
    theTrajectoryCleaner     = new TrajectoryCleanerBySharedHits();          

    //theNavigationSchool      = new SimpleNavigationSchool(&(*theGeomSearchTracker),&(*theMagField));
    edm::ParameterSet NavigationPSet = conf_.getParameter<edm::ParameterSet>("NavigationPSet");
    std::string navigationSchoolName = NavigationPSet.getParameter<std::string>("ComponentName");
    theNavigationSchool = NavigationSchoolFactory::get()->create( navigationSchoolName, &(*theGeomSearchTracker), &(*theMagField));


    // set the correct navigation
    NavigationSetter setter( *theNavigationSchool);

    // set the TrajectoryBuilder
    std::string trajectoryBuilderName = conf_.getParameter<std::string>("TrajectoryBuilder");
    edm::ESHandle<TrackerTrajectoryBuilder> theTrajectoryBuilderHandle;
    es.get<CkfComponentsRecord>().get(trajectoryBuilderName,theTrajectoryBuilderHandle);
    theTrajectoryBuilder = theTrajectoryBuilderHandle.product();    
    
    std::string cleaner = conf_.getParameter<std::string>("RedundantSeedCleaner");
    if (cleaner == "SeedCleanerByHitPosition") {
        theSeedCleaner = new SeedCleanerByHitPosition();
    } else if (cleaner == "SeedCleanerBySharedInput") {
        theSeedCleaner = new SeedCleanerBySharedInput();
    } else if (cleaner == "CachingSeedCleanerByHitPosition") {
        theSeedCleaner = new CachingSeedCleanerByHitPosition();
    } else if (cleaner == "CachingSeedCleanerBySharedInput") {
        theSeedCleaner = new CachingSeedCleanerBySharedInput();
    } else if (cleaner == "none") {
        theSeedCleaner = 0;
    } else {
        throw cms::Exception("RedundantSeedCleaner not found", cleaner);
    }
  }
  
  // Functions that gets called by framework every event
  void CkfTrackCandidateMakerBase::produceBase(edm::Event& e, const edm::EventSetup& es)
  { 
    // method for Debugging
    printHitsDebugger(e);

    // Step A: set Event for the TrajectoryBuilder
    theTrajectoryBuilder->setEvent(e);        
    
    // Step B: Retrieve seeds
    
    std::string seedProducer = conf_.getParameter<std::string>("SeedProducer");
    std::string seedLabel = conf_.getParameter<std::string>("SeedLabel");

    edm::Handle<View<TrajectorySeed> > collseed;
    //edm::Handle<TrajectorySeedCollection> collseed;
    e.getByLabel(seedProducer, seedLabel, collseed);
    //view TrajectorySeedCollection const & theSeedColl = *collseed;
    
    // Step C: Create empty output collection
    std::auto_ptr<TrackCandidateCollection> output(new TrackCandidateCollection);    
    
    
    // Step D: Invoke the building algorithm
    if ((*collseed).size()>0){
      vector<Trajectory> theFinalTrajectories;
      //view TrajectorySeedCollection::const_iterator iseed;
      //view TrajectorySeedCollection::const_iterator end = lastSeed(theSeedColl);

      vector<Trajectory> rawResult;
      if (theSeedCleaner) theSeedCleaner->init( &rawResult );
      
      // method for debugging
      countSeedsDebugger();
      
      size_t collseed_size = collseed->size(); 
      for (size_t j = 0; j < collseed_size; j++){
	//for(iseed=theSeedColl.begin();iseed!=end;iseed++){
	vector<Trajectory> theTmpTrajectories;
      
	if (theSeedCleaner && !theSeedCleaner->good( &((*collseed)[j])) ) continue;
	theTmpTrajectories = theTrajectoryBuilder->trajectories( (*collseed)[j] );
	
       
	LogDebug("CkfPattern") << "======== CkfTrajectoryBuilder returned " << theTmpTrajectories.size()
			       << " trajectories for this seed ========";

	theTrajectoryCleaner->clean(theTmpTrajectories);
      
	for(vector<Trajectory>::const_iterator it=theTmpTrajectories.begin();
	    it!=theTmpTrajectories.end(); it++){
	  if( it->isValid() ) {
	    rawResult.push_back(*it);
            if (theSeedCleaner) theSeedCleaner->add( & (*it) );
	  }
      
	}
	LogDebug("CkfPattern") << "rawResult size after cleaning " << rawResult.size();
      }
      
      if (theSeedCleaner) theSeedCleaner->done();
      
      // Step E: Clean the result
      theTrajectoryCleaner->clean(rawResult);

      vector<Trajectory> & unsmoothedResult(rawResult);
      unsmoothedResult.erase(std::remove_if(unsmoothedResult.begin(),unsmoothedResult.end(),
					    std::not1(std::mem_fun_ref(&Trajectory::isValid))),
			     unsmoothedResult.end());
      

      //      for (vector<Trajectory>::const_iterator itraw = rawResult.begin();
      //	   itraw != rawResult.end(); itraw++) {
      //if((*itraw).isValid()) unsmoothedResult.push_back( *itraw);
      //}

      //analyseCleanedTrajectories(unsmoothedResult);
      

      // Step F: Convert to TrackCandidates
      output->reserve(unsmoothedResult.size());
      for (vector<Trajectory>::const_iterator it = unsmoothedResult.begin();
	   it != unsmoothedResult.end(); it++) {
	
	Trajectory::RecHitContainer thits;
	it->recHitsV(thits);
	OwnVector<TrackingRecHit> recHits;
	recHits.reserve(thits.size());
	for (Trajectory::RecHitContainer::const_iterator hitIt = thits.begin();
	     hitIt != thits.end(); hitIt++) {
	  recHits.push_back( (**hitIt).hit()->clone());
	}
	
	//PTrajectoryStateOnDet state = *(it->seed().startingState().clone());
	std::pair<TrajectoryStateOnSurface, const GeomDet*> initState = 
	  theInitialState->innerState( *it);
      
	// temporary protection againt invalid initial states
	if (! initState.first.isValid() || initState.second == 0) {
          //cout << "invalid innerState, will not make TrackCandidate" << endl;
          continue;
        }

	PTrajectoryStateOnDet* state = TrajectoryStateTransform().persistentState( initState.first,
										   initState.second->geographicalId().rawId());
	//	FitTester fitTester(es);
	//	fitTester.fit( *it);
	
	output->push_back(TrackCandidate(recHits,it->seed(),*state));
	delete state;
      }
      
      
      
      LogTrace("CkfPattern") << "========== CkfTrackCandidateMaker Info ==========";
      edm::ESHandle<TrackerGeometry> tracker;
      es.get<TrackerDigiGeometryRecord>().get(tracker);
      LogTrace("CkfPattern") << "number of Seed: " << collseed->size();
      
      /*
      for(iseed=theSeedColl.begin();iseed!=theSeedColl.end();iseed++){
	DetId tmpId = DetId( iseed->startingState().detId());
	const GeomDet* tmpDet  = tracker->idToDet( tmpId );
	GlobalVector gv = tmpDet->surface().toGlobal( iseed->startingState().parameters().momentum() );
	
	LogTrace("CkfPattern") << "seed perp,phi,eta : " 
				       << gv.perp() << " , " 
				       << gv.phi() << " , " 
				       << gv.eta() ;
      }
      */
      
      LogTrace("CkfPattern") << "number of finalTrajectories: " << unsmoothedResult.size();
      for (vector<Trajectory>::const_iterator it = unsmoothedResult.begin();
	   it != unsmoothedResult.end(); it++) {
	LogTrace("CkfPattern") << "candidate's n valid and invalid hit, chi2, pt : " 
				       << it->foundHits() << " , " 
				       << it->lostHits() <<" , " 
				       <<it->chiSquared() << " , "
				       <<it->lastMeasurement().predictedState().globalMomentum().perp();
      }
      LogTrace("CkfPattern") << "=================================================";
          
    }
    
    // method for debugging
    deleteAssocDebugger();

    // Step G: write output to file
    e.put(output);
  }
}

