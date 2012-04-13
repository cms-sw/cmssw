#include "RecoHI/HiMuonAlgos/interface/DiMuonTrajectorySeed.h"

// construct
namespace cms
{
DiMuonTrajectorySeed::DiMuonTrajectorySeed( const TrajectoryMeasurement& mtm0, const FreeTrajectoryState& ftsmuon, int aMult )
{
                        theFtsMuon=ftsmuon;
                        thePropagationDirection=oppositeToMomentum;
                        theLowMult=aMult;
                        theTrajMeasurements.push_back(mtm0);
                       // theRecHits.push_back(rh->clone()); theDetId = det;
                       // PTraj = boost::shared_ptr<PTrajectoryStateOnDet>(
                       // transformer.persistentState(tsos, theDetId) );
}
}







