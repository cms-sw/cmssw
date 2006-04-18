#include <memory>
#include <string>

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "RecoTracker/CkfPattern/interface/CombinatorialTrackFinder.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"


namespace cms
{

  CombinatorialTrackFinder::CombinatorialTrackFinder(edm::ParameterSet const& conf) : 
    combinatorialTrajectoryBuilder_(conf) ,
    conf_(conf)
  {
    isInitialized = 0;
    cout << "--- CombinatorialTrackFinder::constru is called " << endl;
    produces<TrackCandidateCollection>();  //WHAT'S THIS???
  }


  // Virtual destructor needed.
  CombinatorialTrackFinder::~CombinatorialTrackFinder() { }  

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
      std::vector<Trajectory> theFinalTrajectories;
      TrajectorySeedCollection::const_iterator iseed;
      
      for(iseed=theSeedColl.begin();iseed!=theSeedColl.end();iseed++){
	std::vector<Trajectory> theTmpTrajectories;
	cout << "--- combtrajBuilder.trajectories(seed,ev) called ----- " << endl;
	theTmpTrajectories = combinatorialTrajectoryBuilder_.trajectories(*iseed,e);
	/*
	theFinalTrajectories.insert(theFinalTrajectories.end(),
				    theTmpTrajectories.begin(),
				    theTmpTrajectories.end()    );
	*/
      }
   
     
      // Step D: write output to file
      ///if ((*output).size()>0) e.put(output);
    }
  }
  
}
