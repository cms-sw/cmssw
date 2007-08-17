#include "RecoHIMuon/HiMuSeed/interface/DiMuonTrajectorySeed.h"

// construct
  DiMuonTrajectorySeed::DiMuonTrajectorySeed( 
                        TrajectoryStateOnSurface tsos, 
                        const FreeTrajectoryState& ftsmuon, 
			const TrackingRecHit*  rh, 
			int aMult,
			int det   
                        )
  {
                        theFtsMuon = ftsmuon;
		        thePropagationDirection = oppositeToMomentum;
		        theLowMult = aMult;
			theRecHits.push_back(rh->clone());
			theDetId = det;
			std::cout<< " DiMuonTrajectorySeed::Point 1 "<<std::endl;
  PTraj = boost::shared_ptr<PTrajectoryStateOnDet>(
                      transformer.persistentState(tsos, theDetId) );
			std::cout<< " DiMuonTrajectorySeed::Point 2 "<<std::endl;
			
  }








