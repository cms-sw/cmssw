
#include "DataFormats/EgammaReco/interface/ElectronNHitSeed.h"


using namespace reco ;


ElectronNHitSeed::ElectronNHitSeed()
  : TrajectorySeed(), ctfTrack_(), caloCluster_(), hitInfo_(),
    nrLayersAlongTraj_(0),
    isEcalDriven_(false), isTrackerDriven_(false)
   
{}

ElectronNHitSeed::ElectronNHitSeed
( const TrajectorySeed & seed )
  : TrajectorySeed(seed),
    ctfTrack_(), caloCluster_(), hitInfo_(),
    nrLayersAlongTraj_(0),
    isEcalDriven_(false), isTrackerDriven_(false)
{}

ElectronNHitSeed::ElectronNHitSeed
 ( PTrajectoryStateOnDet & pts, recHitContainer & rh, PropagationDirection & dir )
   : TrajectorySeed(pts,rh,dir),
     ctfTrack_(), caloCluster_(), hitInfo_(),
     nrLayersAlongTraj_(0),
     isEcalDriven_(false), isTrackerDriven_(false)
{}

void ElectronNHitSeed::setCtfTrack
( const CtfTrackRef & ctfTrack )
{
  ctfTrack_ = ctfTrack ;
  isTrackerDriven_ = true ;
}

