
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/TrackReco/interface/Track.h"

using namespace reco ;

ElectronSeed::ElectronSeed()
 : TrajectorySeed(), ctfTrack_(), caloCluster_(),
   isEcalDriven_(false), isTrackerDriven_(false),
   subDet2_(0), dRz2_(std::numeric_limits<float>::infinity()),
   dPhi2_(std::numeric_limits<float>::infinity())

 {}

ElectronSeed::ElectronSeed
 ( const TrajectorySeed & seed )
 : TrajectorySeed(seed),
   ctfTrack_(), caloCluster_(),
   isEcalDriven_(false), isTrackerDriven_(false),
   subDet2_(0), dRz2_(std::numeric_limits<float>::infinity()),
   dPhi2_(std::numeric_limits<float>::infinity())
 {}

ElectronSeed::ElectronSeed
 ( PTrajectoryStateOnDet & pts, recHitContainer & rh, PropagationDirection & dir )
 : TrajectorySeed(pts,rh,dir),
   ctfTrack_(), caloCluster_(),
   isEcalDriven_(false), isTrackerDriven_(false),
   subDet2_(0), dRz2_(std::numeric_limits<float>::infinity()),
   dPhi2_(std::numeric_limits<float>::infinity())
 {}

void ElectronSeed::setCtfTrack
 ( const CtfTrackRef & ctfTrack )
 { 
  ctfTrack_ = ctfTrack ; 
  isTrackerDriven_ = true ;
 }

void ElectronSeed::setCaloCluster
 ( const CaloClusterRef & scl,
   int subDet2, float dRz2, float dPhi2 )
 {
  caloCluster_ = scl ;
  isEcalDriven_ = true ;
  subDet2_ = subDet2 ;
  dRz2_ = dRz2 ;
  dPhi2_ = dPhi2 ;
 }

ElectronSeed::~ElectronSeed()
 {}

