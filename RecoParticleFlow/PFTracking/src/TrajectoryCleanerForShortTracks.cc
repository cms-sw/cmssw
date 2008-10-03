#include "RecoParticleFlow/PFTracking/interface/TrajectoryCleanerForShortTracks.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/RecHitComparatorByPosition.h"
#include <map>
#include <vector>

using namespace std;

void TrajectoryCleanerForShortTracks::clean( TrajectoryPointerContainer & tc) const
{
  typedef vector<Trajectory*>::iterator TI;
  typedef map<const TransientTrackingRecHit*,vector<TI>,RecHitComparatorByPosition> RecHitMap; 
  typedef map<Trajectory*,int,less<Trajectory*> > TrajMap;  // for each Trajectory it stores the number of shared hits
  RecHitMap theRecHitMap;
 
  // start filling theRecHit
  for (TrajectoryCleaner::TrajectoryPointerIterator
	 it = tc.begin(); it != tc.end(); it++) {
    Trajectory::DataContainer pd = (*it)->data();
    for (Trajectory::DataContainer::iterator im = pd.begin();
    	 im != pd.end(); im++) {
      //RC const TransientTrackingRecHit* theRecHit = ((*im).recHit());
      const TransientTrackingRecHit* theRecHit = &(*(*im).recHit());
      if (theRecHit->isValid())
        theRecHitMap[theRecHit].push_back(it);
    }
  }
  // end filling theRecHit

  // for each trajectory fill theTrajMap

  for (TrajectoryCleaner::TrajectoryPointerIterator
	 itt = tc.begin(); itt != tc.end(); itt++) {
 
    if((*itt)->isValid()){  
      TrajMap theTrajMap;
      Trajectory::DataContainer pd = (*itt)->data();
      for (Trajectory::DataContainer::iterator im = pd.begin();
	   im != pd.end(); im++) {
	//RC const TransientTrackingRecHit* theRecHit = ((*im).recHit());
	const TransientTrackingRecHit* theRecHit = &(*(*im).recHit());
        if (theRecHit->isValid()) {
	  const vector<TI>& hitTrajectories( theRecHitMap[theRecHit]);
	  for (vector<TI>::const_iterator ivec=hitTrajectories.begin(); 
	       ivec!=hitTrajectories.end(); ivec++) {
	    if (*ivec != itt){
	      if ((**ivec)->isValid()){
		theTrajMap[**ivec]++;
	      }
	    }
	  }
	}
      }
      //end filling theTrajMap

      // check for duplicated tracks
      if(!theTrajMap.empty() > 0){
	for(TrajMap::iterator imapp = theTrajMap.begin(); 
	    imapp != theTrajMap.end(); imapp++){
	  //          int nhit1 = (*itt).data().size();
	  //          int nhit2 = (*imapp).first->data().size();
          int nhit1 = (*itt)->foundHits();
          int nhit2 = (*imapp).first->foundHits();

	  
	  if((*imapp).second >= min(nhit1, nhit2)/2.){
	    Trajectory* badtraj;
	    if (nhit1 != nhit2)
	      // select the shortest trajectory
	      badtraj = (nhit1 > nhit2) ?
		(*imapp).first : *itt;
	    else
	      // select the trajectory with less chi squared
	      badtraj = ((*imapp).first->chiSquared() > (*itt)->chiSquared()) ?
		(*imapp).first : *itt;
	    badtraj->invalidate();  // invalidate this trajectory
	  }

	}
      } 
    }
  }

}

