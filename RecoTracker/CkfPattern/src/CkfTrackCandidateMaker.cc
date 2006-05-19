#include <memory>
#include <string>

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "RecoTracker/CkfPattern/interface/CkfTrackCandidateMaker.h"

#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
//#include "RecoTracker/CkfPattern/interface/FitTester.h"

using namespace edm;


namespace cms{
  CkfTrackCandidateMaker::CkfTrackCandidateMaker(edm::ParameterSet const& conf) : 
    conf_(conf),theCkfTrajectoryBuilder(0),theTrajectoryCleaner(0),
    theInitialState(0),theMeasurementTracker(0),theNavigationSchool(0)
  {  
    produces<TrackCandidateCollection>();  
  }

  
  // Virtual destructor needed.
  CkfTrackCandidateMaker::~CkfTrackCandidateMaker() {
    delete theInitialState;
    delete theMeasurementTracker;
    delete theNavigationSchool;
    delete theCkfTrajectoryBuilder;
    delete theTrajectoryCleaner;    
  }  

  void CkfTrackCandidateMaker::beginJob (EventSetup const & es)
  {
    //services
    es.get<TrackerRecoGeometryRecord>().get( theGeomSearchTracker );
    es.get<IdealMagneticFieldRecord>().get(theMagField);
      
    theInitialState       = new TransientInitialStateEstimator( es);
    
    // get nested parameter set for the MeasurementTracker
    ParameterSet mt_params = conf_.getParameter<ParameterSet>("MeasurementTrackerParameters") ;
    theMeasurementTracker = new MeasurementTracker(es, mt_params);

    theNavigationSchool   = new SimpleNavigationSchool(&(*theGeomSearchTracker),&(*theMagField));
      
    // set the correct navigation
    NavigationSetter setter( *theNavigationSchool);

    theCkfTrajectoryBuilder = new CkfTrajectoryBuilder(conf_,es,theMeasurementTracker);
    theTrajectoryCleaner = new TrajectoryCleanerBySharedHits();    
  }
  
  // Functions that gets called by framework every event
  void CkfTrackCandidateMaker::produce(edm::Event& e, const edm::EventSetup& es)
  {        
    // Step A: update MeasurementTracker
    theMeasurementTracker->update(e);
        
    
    // Step B: Retrieve seeds
    
    std::string seedProducer = conf_.getParameter<std::string>("SeedProducer");
    edm::Handle<TrajectorySeedCollection> collseed;
    e.getByLabel(seedProducer, collseed);
    //    e.getByType(collseed);
    TrajectorySeedCollection theSeedColl = *collseed;
    
    // Step C: Create empty output collection
    std::auto_ptr<TrackCandidateCollection> output(new TrackCandidateCollection);    
    
    
    // Step D: Invoke the building algorithm
    if ((*collseed).size()>0){
      vector<Trajectory> theFinalTrajectories;
      TrajectorySeedCollection::const_iterator iseed;
      
      vector<Trajectory> rawResult;
      for(iseed=theSeedColl.begin();iseed!=theSeedColl.end();iseed++){
	vector<Trajectory> theTmpTrajectories;
	theTmpTrajectories = theCkfTrajectoryBuilder->trajectories(*iseed,e);
	
	cout << "CkfTrajectoryBuilder returned " << theTmpTrajectories.size()
	     << " trajectories" << endl;

	theTrajectoryCleaner->clean(theTmpTrajectories);
      
	for(vector<Trajectory>::const_iterator it=theTmpTrajectories.begin();
	    it!=theTmpTrajectories.end(); it++){
	  if( it->isValid() ) {
	    rawResult.push_back(*it);
	  }
	}
	cout << "rawResult size after cleaning " << rawResult.size() << endl;
      }
      
      // Step E: Clean the result
      vector<Trajectory> unsmoothedResult;
      theTrajectoryCleaner->clean(rawResult);
      
      for (vector<Trajectory>::const_iterator itraw = rawResult.begin();
	   itraw != rawResult.end(); itraw++) {
	if((*itraw).isValid()) unsmoothedResult.push_back( *itraw);
      }
      //analyseCleanedTrajectories(unsmoothedResult);
      

      // Step F: Convert to TrackCandidates
      for (vector<Trajectory>::const_iterator it = unsmoothedResult.begin();
	   it != unsmoothedResult.end(); it++) {
	
	OwnVector<TrackingRecHit> recHits;
	OwnVector<TransientTrackingRecHit> thits = it->recHits();
	for (OwnVector<TransientTrackingRecHit>::const_iterator hitIt = thits.begin(); 
	     hitIt != thits.end(); hitIt++) {
	  recHits.push_back( hitIt->hit()->clone());
	}
	
	TrajectorySeed seed         = *(it->seed().clone());

	//PTrajectoryStateOnDet state = *(it->seed().startingState().clone());
	std::pair<TrajectoryStateOnSurface, const GeomDet*> initState = 
	  theInitialState->innerState( *it);
	PTrajectoryStateOnDet* state = TrajectoryStateTransform().persistentState( initState.first,
										   initState.second->geographicalId().rawId());
	//	FitTester fitTester(es);
	//	fitTester.fit( *it);
	
	output->push_back(TrackCandidate(recHits,seed,*state));
      }
      
      
      
      cout << " ========== DEBUG CkfTrackCandidateMaker: start  ========== " << endl;
      edm::ESHandle<TrackerGeometry> tracker;
      es.get<TrackerDigiGeometryRecord>().get(tracker);
      cout << "number of Seed: " << theSeedColl.size() << endl;
      
      /*
	for(iseed=theSeedColl.begin();iseed!=theSeedColl.end();iseed++){
	DetId tmpId = DetId( iseed->startingState().detId());
	const GeomDet* tmpDet  = tracker->idToDet( tmpId );
	GlobalVector gv = tmpDet->surface().toGlobal( iseed->startingState().parameters().momentum() );
	
	cout << "seed perp,phi,eta : " 
	<< gv.perp() << " , " 
	<< gv.phi() << " , " 
	<< gv.eta() << endl;
	}
      */
      
      cout << "number of finalTrajectories: " << unsmoothedResult.size() << endl;
      for (vector<Trajectory>::const_iterator it = unsmoothedResult.begin();
	   it != unsmoothedResult.end(); it++) {
	cout << "n valid and invalid hit, chi2 : " 
	     << it->foundHits() << " , " << it->lostHits() <<" , " <<it->chiSquared() << endl;
      }
      cout << " ========== DEBUG CkfTrackCandidateMaker: end ========== " << endl;
      

      
      // Step G: write output to file
      e.put(output);
    }
  }
}

