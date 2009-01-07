
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/TrackReco/interface/Track.h"

using namespace reco ;

ElectronSeed::ElectronSeed()
 : TrajectorySeed(), ctfTrack_(), caloCluster_(),
   subDet2_(0), dRz2_(std::numeric_limits<float>::infinity()),
   dPhi2_(std::numeric_limits<float>::infinity())

 {}

ElectronSeed::ElectronSeed
 ( const TrajectorySeed & seed, const CtfTrackRef & ctfTrack )
 : TrajectorySeed(seed),
   ctfTrack_(ctfTrack), caloCluster_(),
   subDet2_(0), dRz2_(std::numeric_limits<float>::infinity()),
   dPhi2_(std::numeric_limits<float>::infinity())
 {}

ElectronSeed::ElectronSeed
 ( const TrajectorySeed & seed, const CaloClusterRef & scl,
   int subDet2, float dRz2, float dPhi2 )
 : TrajectorySeed(seed),
   ctfTrack_(), caloCluster_(scl),
   subDet2_(subDet2), dRz2_(dRz2), dPhi2_(dPhi2)
 {}

ElectronSeed::ElectronSeed
 ( PTrajectoryStateOnDet & pts, recHitContainer & rh, PropagationDirection & dir,
   const CaloClusterRef & scl, int subDet2, float dRz2, float dPhi2 )
 : TrajectorySeed(pts,rh,dir),
   ctfTrack_(), caloCluster_(scl),
   subDet2_(subDet2), dRz2_(dRz2), dPhi2_(dPhi2)
 {}

ElectronSeed::~ElectronSeed()
 {}

