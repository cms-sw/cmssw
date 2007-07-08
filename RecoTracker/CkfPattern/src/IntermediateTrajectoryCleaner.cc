#include "RecoTracker/CkfPattern/interface/IntermediateTrajectoryCleaner.h"
//#define GIO_WAIT
#ifdef GIO_WAIT
TempTrajectoryContainer
IntermediateTrajectoryCleaner::clean(TempTrajectoryContainer &theTrajectories) {
        return theTrajectories;
}
#else 
IntermediateTrajectoryCleaner::TempTrajectoryContainer
IntermediateTrajectoryCleaner::clean(IntermediateTrajectoryCleaner::TempTrajectoryContainer &theTrajectories) {

  if (theTrajectories.empty()) return TempTrajectoryContainer();
  if (theTrajectories[0].measurements().size()<4) return TempTrajectoryContainer(theTrajectories);

  TempTrajectoryContainer result;

  for (TempTrajectoryContainer::iterator firstTraj=theTrajectories.begin(), firstEnd=theTrajectories.end() - 1;
     firstTraj != firstEnd; ++firstTraj) {

    if ( (!firstTraj->isValid()) ||
         (!firstTraj->lastMeasurement().recHit()->isValid()) ) continue;
    
    TempTrajectory::DataContainer::const_iterator itFirst = firstTraj->measurements().rbegin();
    ConstRecHitPointer first_hit1 = itFirst->recHit(); --itFirst; 
    ConstRecHitPointer first_hit2 = itFirst->recHit(); --itFirst; 
    ConstRecHitPointer first_hit3 = itFirst->recHit();

    for (TempTrajectoryContainer::iterator secondTraj = (firstTraj+1), secondEnd = theTrajectories.end();
       secondTraj != secondEnd; ++secondTraj) {

      if ( (!secondTraj->isValid()) ||
           (!secondTraj->lastMeasurement().recHit()->isValid()) ) continue;

        TempTrajectory::DataContainer::const_iterator itSecond = secondTraj->measurements().rbegin();
        ConstRecHitPointer second_hit1 = itSecond->recHit(); --itSecond; 
        ConstRecHitPointer second_hit2 = itSecond->recHit(); --itSecond; 
        ConstRecHitPointer second_hit3 = itSecond->recHit();

      if ( first_hit2->isValid() ^ second_hit2->isValid()) { // ^ = XOR!
        if ( first_hit1->hit()->sharesInput( second_hit1->hit(), TrackingRecHit::all ) ){
          if ( &(*first_hit3) == &(*second_hit3) ) {

            if (second_hit2->isValid()) {
              firstTraj->invalidate();
              break;
            }
            //if (!second_hit2->isValid())  // should be always true, as we did XOR !
            secondTraj->invalidate();
          }
        }
      }
    }
  }

  for (TempTrajectoryContainer::const_iterator it = theTrajectories.begin(), end = theTrajectories.end(); it != end; ++it) {
    if((*it).isValid()) result.push_back( *it);
  }

  return result;
}
#endif
