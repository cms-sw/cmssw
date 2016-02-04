
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/TrackReco/interface/Track.h"

using namespace reco ;


ElectronSeed::ElectronSeed()
 : TrajectorySeed(), ctfTrack_(), caloCluster_(), hitsMask_(0),
   subDet2_(0),
   dRz2_(std::numeric_limits<float>::infinity()),
   dPhi2_(std::numeric_limits<float>::infinity()),
   dRz2Pos_(std::numeric_limits<float>::infinity()),
   dPhi2Pos_(std::numeric_limits<float>::infinity()),
   subDet1_(0),
   dRz1_(std::numeric_limits<float>::infinity()),
   dPhi1_(std::numeric_limits<float>::infinity()),
   dRz1Pos_(std::numeric_limits<float>::infinity()),
   dPhi1Pos_(std::numeric_limits<float>::infinity()),
   hcalDepth1OverEcal_(std::numeric_limits<float>::infinity()),
   hcalDepth2OverEcal_(std::numeric_limits<float>::infinity()),
   isEcalDriven_(false), isTrackerDriven_(false)
 {}

ElectronSeed::ElectronSeed
 ( const TrajectorySeed & seed )
 : TrajectorySeed(seed),
   ctfTrack_(), caloCluster_(), hitsMask_(0),
   subDet2_(0),
   dRz2_(std::numeric_limits<float>::infinity()),
   dPhi2_(std::numeric_limits<float>::infinity()),
   dRz2Pos_(std::numeric_limits<float>::infinity()),
   dPhi2Pos_(std::numeric_limits<float>::infinity()),
   subDet1_(0),
   dRz1_(std::numeric_limits<float>::infinity()),
   dPhi1_(std::numeric_limits<float>::infinity()),
   dRz1Pos_(std::numeric_limits<float>::infinity()),
   dPhi1Pos_(std::numeric_limits<float>::infinity()),
   hcalDepth1OverEcal_(std::numeric_limits<float>::infinity()),
   hcalDepth2OverEcal_(std::numeric_limits<float>::infinity()),
   isEcalDriven_(false), isTrackerDriven_(false)
 {}

ElectronSeed::ElectronSeed
 ( PTrajectoryStateOnDet & pts, recHitContainer & rh, PropagationDirection & dir )
 : TrajectorySeed(pts,rh,dir),
   ctfTrack_(), caloCluster_(), hitsMask_(0),
   subDet2_(0),
   dRz2_(std::numeric_limits<float>::infinity()),
   dPhi2_(std::numeric_limits<float>::infinity()),
   dRz2Pos_(std::numeric_limits<float>::infinity()),
   dPhi2Pos_(std::numeric_limits<float>::infinity()),
   subDet1_(0),
   dRz1_(std::numeric_limits<float>::infinity()),
   dPhi1_(std::numeric_limits<float>::infinity()),
   dRz1Pos_(std::numeric_limits<float>::infinity()),
   dPhi1Pos_(std::numeric_limits<float>::infinity()),
   hcalDepth1OverEcal_(std::numeric_limits<float>::infinity()),
   hcalDepth2OverEcal_(std::numeric_limits<float>::infinity()),
   isEcalDriven_(false), isTrackerDriven_(false)
 {}

void ElectronSeed::setCtfTrack
 ( const CtfTrackRef & ctfTrack )
 {
  ctfTrack_ = ctfTrack ;
  isTrackerDriven_ = true ;
 }

void ElectronSeed::setCaloCluster
 ( const CaloClusterRef & scl,
   unsigned char hitsMask,
   int subDet2, int subDet1,
   float hoe1, float hoe2 )
 {
  caloCluster_ = scl ;
  hitsMask_ = hitsMask ;
  isEcalDriven_ = true ;
  subDet2_ = subDet2 ;
  subDet1_ = subDet1 ;
  hcalDepth1OverEcal_ = hoe1 ;
  hcalDepth2OverEcal_ = hoe2 ;
 }

void ElectronSeed::setNegAttributes
 ( float dRz2, float dPhi2, float dRz1, float dPhi1 )
 {
  dRz2_ = dRz2 ;
  dPhi2_ = dPhi2 ;
  dRz1_ = dRz1 ;
  dPhi1_ = dPhi1 ;
 }

void ElectronSeed::setPosAttributes
 ( float dRz2, float dPhi2, float dRz1, float dPhi1 )
 {
  dRz2Pos_ = dRz2 ;
  dPhi2Pos_ = dPhi2 ;
  dRz1Pos_ = dRz1 ;
  dPhi1Pos_ = dPhi1 ;
 }

ElectronSeed::~ElectronSeed()
 {}

