
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"


using namespace reco ;


ElectronSeed::ElectronSeed()
  : TrajectorySeed(), ctfTrack_(), caloCluster_(), hitInfo_(),
    nrLayersAlongTraj_(0),
    isEcalDriven_(false), isTrackerDriven_(false)
   
{}

ElectronSeed::ElectronSeed
( const TrajectorySeed & seed )
  : TrajectorySeed(seed),
    ctfTrack_(), caloCluster_(), hitInfo_(),
    nrLayersAlongTraj_(0),
    isEcalDriven_(false), isTrackerDriven_(false)
{}

ElectronSeed::ElectronSeed
 ( PTrajectoryStateOnDet & pts, recHitContainer & rh, PropagationDirection & dir )
   : TrajectorySeed(pts,rh,dir),
     ctfTrack_(), caloCluster_(), hitInfo_(),
     nrLayersAlongTraj_(0),
     isEcalDriven_(false), isTrackerDriven_(false)
{}

void ElectronSeed::setCtfTrack
( const CtfTrackRef & ctfTrack )
{
  ctfTrack_ = ctfTrack ;
  isTrackerDriven_ = true ;
}

//the hit mask tells us which hits were used in the seed
//typically all are used but this could change in the future
int ElectronSeed::hitsMask()const
{
  int mask=0;
  for(size_t hitNr=0;hitNr<nHits();hitNr++){
    int bitNr = 0x1 << hitNr;
    int hitDetId = (recHits().first+hitNr)->geographicalId().rawId();
    auto detIdMatcher = [hitDetId](const ElectronSeed::PMVars& var){return hitDetId==var.detId;};
    if(std::find_if(hitInfo_.begin(),hitInfo_.end(),detIdMatcher)!=hitInfo_.end()){
      mask|=bitNr;
    }
  }
  return mask;
}

void ElectronSeed::setNegAttributes(float dRZ2,float dPhi2,float dRZ1,float dPhi1)
{
  if(hitInfo_.empty()) hitInfo_.resize(2);
  if(hitInfo_.size()!=2){
    throw cms::Exception("LogicError") <<"ElectronSeed::setNegAttributes should only operate on seeds with exactly two hits. This is because it is a legacy function to preverse backwards compatiblity and should not be used on new code which matches variable number of hits";
  }
  hitInfo_[0].dRZNeg = dRZ1;
  hitInfo_[1].dRZNeg = dRZ2;
  hitInfo_[0].dPhiNeg = dPhi1;
  hitInfo_[1].dPhiNeg = dPhi2;
  
}

void ElectronSeed::setPosAttributes(float dRZ2,float dPhi2,float dRZ1,float dPhi1)
{
  if(hitInfo_.empty()) hitInfo_.resize(2);
  if(hitInfo_.size()!=2){
    throw cms::Exception("LogicError") <<"ElectronSeed::setPosAttributes should only operate on seeds with exactly two hits. This is because it is a legacy function to preverse backwards compatiblity and should not be used on new code which matches variable number of hits";
  }
  hitInfo_[0].dRZPos = dRZ1;
  hitInfo_[1].dRZPos = dRZ2;
  hitInfo_[0].dPhiPos = dPhi1;
  hitInfo_[1].dPhiPos = dPhi2;
}

ElectronSeed::PMVars::PMVars():
  dRZPos(std::numeric_limits<float>::infinity()),
  dRZNeg(std::numeric_limits<float>::infinity()),
  dPhiPos(std::numeric_limits<float>::infinity()),
  dPhiNeg(std::numeric_limits<float>::infinity()),
  detId(0),
  layerOrDiskNr(-1)
{}
