#include <memory>
#include <string>

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "RecoTracker/CkfPattern/interface/CombinatorialTrackFinder.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"

// --- temporary.for debug
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
// ---
namespace cms
{

  CombinatorialTrackFinder::CombinatorialTrackFinder(edm::ParameterSet const& conf) : 
    combinatorialTrajectoryBuilder_(conf) ,
    conf_(conf)
  {
    isInitialized = 0;
    cout << "--- CombinatorialTrackFinder::constru is called " << endl;
    theTrajectoryCleaner = new TrajectoryCleanerBySharedHits();
    produces<TrackCandidateCollection>();  //WHAT'S THIS???
  }


  // Virtual destructor needed.
  CombinatorialTrackFinder::~CombinatorialTrackFinder() {
    delete theTrajectoryCleaner;
  }  

  // Functions that gets called by framework every event
  void CombinatorialTrackFinder::produce(edm::Event& e, const edm::EventSetup& es)
  {    
    // Bad temporary solution!!!
    if(isInitialized == 0){
      combinatorialTrajectoryBuilder_.init(es);
      isInitialized = 1;
    }

    cout << endl << "==== CombinatorialTrackFinder::produce is called ==== " << endl;
    
    // retrieve seeds
    edm::Handle<TrajectorySeedCollection> collseed;
    e.getByType(collseed);
    
    TrajectorySeedCollection theSeedColl = *collseed;
    cout << "--- seeds are got --- " << endl;

    // Step B: create empty output collection
    std::auto_ptr<TrackCandidateCollection> output(new TrackCandidateCollection);
    


    // Step C: Invoke the building algorithm
    if ((*collseed).size()>0){
      cout << "---- collseed->size(): " << collseed->size() << " ------" << endl;
      vector<Trajectory> theFinalTrajectories;
      TrajectorySeedCollection::const_iterator iseed;
            
      vector<Trajectory> rawResult;
      for(iseed=theSeedColl.begin();iseed!=theSeedColl.end();iseed++){
	vector<Trajectory> theTmpTrajectories;
	theTmpTrajectories = combinatorialTrajectoryBuilder_.trajectories(*iseed,e);
	
	theTrajectoryCleaner->clean(theTmpTrajectories);
	
	for(vector<Trajectory>::const_iterator it=theTmpTrajectories.begin();
	    it!=theTmpTrajectories.end(); it++){
	  if( it->isValid() ) rawResult.push_back(*it);
	}
      }
      
      // clean the result
      vector<Trajectory> unsmoothedResult;
      theTrajectoryCleaner->clean(rawResult);
      
      for (vector<Trajectory>::const_iterator itraw = rawResult.begin();
	   itraw != rawResult.end(); itraw++) {
	if((*itraw).isValid()) unsmoothedResult.push_back( *itraw);
      }
      
      cout << " ========== DEBUG TRACKFINDER: start  ========== " << endl;
      edm::ESHandle<TrackerGeometry> tracker;
      es.get<TrackerDigiGeometryRecord>().get(tracker);
      for(iseed=theSeedColl.begin();iseed!=theSeedColl.end();iseed++){
	DetId tmpId = DetId( iseed->startingState().detId());
	const GeomDet* tmpDet  = tracker->idToDet( tmpId );
	GlobalVector gv = tmpDet->surface().toGlobal( iseed->startingState().parameters().momentum() );
	
	cout << "seed perp,phi,eta : " 
	     << gv.perp() << " , " 
	     << gv.phi() << " , " 
	     << gv.eta() << endl;
      }
      
      cout << "numb finalTraj: " << unsmoothedResult.size() << endl;
      for (vector<Trajectory>::const_iterator it = unsmoothedResult.begin();
	   it != unsmoothedResult.end(); it++) {
	cout << "n hit valid and invalid: " 
	     << it->foundHits() << " , " << it->lostHits() << endl;
      }
      cout << " ========== DEBUG TRACKFINDER: end ========== " << endl;
      
      // Step D: write output to file
      ///if ((*output).size()>0) e.put(output);
    }
  }
  
}
