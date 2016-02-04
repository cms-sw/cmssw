#include "RecoTracker/CkfPattern/interface/IntermediateTrajectoryCleaner.h"
#include <boost/bind.hpp>
#include <algorithm>
#include <functional>

//#define GIO_WAIT
#ifdef GIO_WAIT
void
IntermediateTrajectoryCleaner::clean(TempTrajectoryContainer &theTrajectories) {
}
#else 

void
IntermediateTrajectoryCleaner::clean(IntermediateTrajectoryCleaner::TempTrajectoryContainer &theTrajectories) {

  if (theTrajectories.empty()) return;
  if (theTrajectories[0].measurements().size()<4) return;

  for (TempTrajectoryContainer::iterator firstTraj=theTrajectories.begin(), firstEnd=theTrajectories.end() - 1;
     firstTraj != firstEnd; ++firstTraj) {

    if ( (!firstTraj->isValid()) ||
         (!firstTraj->lastMeasurement().recHit()->isValid()) ) continue;
    
    TempTrajectory::DataContainer::const_iterator itFirst = firstTraj->measurements().rbegin();
    ConstRecHitPointer first_hit1 = itFirst->recHit(); --itFirst; 
    ConstRecHitPointer first_hit2 = itFirst->recHit(); --itFirst; 
    ConstRecHitPointer first_hit3 = itFirst->recHit();

    bool fh2Valid = first_hit2->isValid();

    for (TempTrajectoryContainer::iterator secondTraj = (firstTraj+1), secondEnd = theTrajectories.end();
       secondTraj != secondEnd; ++secondTraj) {

      if ( (!secondTraj->isValid()) ||
           (!secondTraj->lastMeasurement().recHit()->isValid()) ) continue;

        TempTrajectory::DataContainer::const_iterator itSecond = secondTraj->measurements().rbegin();
        ConstRecHitPointer second_hit1 = itSecond->recHit(); --itSecond; 
        ConstRecHitPointer second_hit2 = itSecond->recHit(); --itSecond; 
        ConstRecHitPointer second_hit3 = itSecond->recHit();

	if ( &(*first_hit3) == &(*second_hit3) ) {
	  if (fh2Valid ^ second_hit2->isValid()) { // ^ = XOR!
	    if ( first_hit1->hit()->sharesInput( second_hit1->hit(), TrackingRecHit::all ) ){
	      
	      if (!fh2Valid) {
		firstTraj->invalidate();
		break;
	      }
	      // else // if (!second_hit2->isValid())  // should be always true, as we did XOR !
	      secondTraj->invalidate();
	    }
	  }
	}
    }
  }
  theTrajectories.erase(std::remove_if( theTrajectories.begin(),theTrajectories.end(),
					std::not1(std::mem_fun_ref(&TempTrajectory::isValid))),
 //					boost::bind(&TempTrajectory::isValid,_1)), 
			theTrajectories.end());
}
#endif
