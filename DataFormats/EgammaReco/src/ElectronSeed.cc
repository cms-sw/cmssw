
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"

#include <climits>

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

ElectronSeed::~ElectronSeed()=default;

void ElectronSeed::setCtfTrack
( const CtfTrackRef & ctfTrack )
{
  ctfTrack_ = ctfTrack ;
  isTrackerDriven_ = true ;
}

//the hit mask tells us which hits were used in the seed
//typically all are used at the HLT but this could change in the future
//RECO only uses some of them
unsigned int ElectronSeed::hitsMask()const
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

void ElectronSeed::initTwoHitSeed(const unsigned char hitMask)
{
  hitInfo_.resize(2);

  std::vector<unsigned int> hitNrs = hitNrsFromMask(hitMask);
  if(hitNrs.size()!=2){
    throw cms::Exception("LogicError") 
      <<"in ElectronSeed::"<<__FUNCTION__<<","<<__LINE__
      <<": number of hits in hit mask is "<<hitNrs.size()<<"\n"
      <<"pre-2017 pixel upgrade ecalDriven ElectronSeeds should have exactly 2 hits\n "
      <<"mask "<<static_cast<unsigned int>(hitMask)<<std::endl;
  }
  if(hitNrs[0]>=nHits() || hitNrs[1]>=nHits()){
    throw cms::Exception("LogicError")
      <<"in ElectronSeed::"<<__FUNCTION__<<","<<__LINE__
      <<": hits are "<<hitNrs[0]<<" and  "<<hitNrs[1]
      <<" while number of hits are "<<nHits()<<"\n"
      <<"this means there was a bug in storing or creating the electron seeds "
      <<"mask "<<static_cast<unsigned int>(hitMask)<<std::endl;
  }
  for(size_t hitNr=0;hitNr<hitInfo_.size();hitNr++){
    auto& info = hitInfo_[hitNr];
    info.setDPhi(std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity());
    info.setDRZ(std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity());
    info.setDet((recHits().first+hitNrs[hitNr])->geographicalId(),-1);
  }
  
}

void ElectronSeed::setNegAttributes(float dRZ2,float dPhi2,float dRZ1,float dPhi1)
{
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
  if(hitInfo_.size()!=2){
    throw cms::Exception("LogicError") <<"ElectronSeed::setPosAttributes should only operate on seeds with exactly two hits. This is because it is a legacy function to preverse backwards compatiblity and should not be used on new code which matches variable number of hits";
  }
  hitInfo_[0].dRZPos = dRZ1;
  hitInfo_[1].dRZPos = dRZ2;
  hitInfo_[0].dPhiPos = dPhi1;
  hitInfo_[1].dPhiPos = dPhi2;
}

std::vector<unsigned int>
ElectronSeed::hitNrsFromMask(unsigned int hitMask)
{
  std::vector<unsigned int> hitNrs;
  for(size_t bitNr=0; bitNr<sizeof(hitMask)*CHAR_BIT ; bitNr++){
    char bit = 0x1 << bitNr;
    if((hitMask&bit)!=0) hitNrs.push_back(bitNr);
  }
  return hitNrs;
}

std::vector<ElectronSeed::PMVars> 
ElectronSeed::createHitInfo(const float dPhi1Pos,const float dPhi1Neg,
			    const float dRZ1Pos,const float dRZ1Neg,
			    const float dPhi2Pos,const float dPhi2Neg,
			    const float dRZ2Pos,const float dRZ2Neg,
			    const char hitMask,const TrajectorySeed::range recHits)
{
  if(hitMask==0) return std::vector<ElectronSeed::PMVars>(); //was trackerDriven so no matched hits

  size_t nrRecHits = std::distance(recHits.first,recHits.second);
  std::vector<unsigned int> hitNrs = hitNrsFromMask(hitMask);

  if(hitNrs.size()!=2){
    throw cms::Exception("LogicError")
      <<"in ElectronSeed::"<<__FUNCTION__<<","<<__LINE__
      <<": number of hits in hit mask is "<<nrRecHits<<"\n"
      <<"pre-2017 pixel upgrade ecalDriven ElectronSeeds should have exactly 2 hits\n"
      <<"mask "<<static_cast<unsigned int>(hitMask)<<std::endl;
  }
  if(hitNrs[0]>=nrRecHits || hitNrs[1]>=nrRecHits){
    throw cms::Exception("LogicError")
      <<"in ElectronSeed::"<<__FUNCTION__<<","<<__LINE__
      <<": hits are "<<hitNrs[0]<<" and "<<hitNrs[1]
      <<" while number of hits are "<<nrRecHits<<"\n"
      <<"this means there was a bug in storing or creating the electron seeds "
      <<"mask "<<static_cast<unsigned int>(hitMask)<<std::endl;
  }
  
  std::vector<PMVars> hitInfo(2);
  hitInfo[0].setDPhi(dPhi1Pos,dPhi1Neg);
  hitInfo[0].setDRZ(dRZ1Pos,dRZ1Neg);
  hitInfo[0].setDet((recHits.first+hitNrs[0])->geographicalId(),-1); //getting the layer information needs tracker topo, hence why its stored in the first as its a pain to access
  hitInfo[1].setDPhi(dPhi2Pos,dPhi2Neg);
  hitInfo[1].setDRZ(dRZ2Pos,dRZ2Neg);
  hitInfo[1].setDet((recHits.first+hitNrs[1])->geographicalId(),-1); //getting the layer information needs tracker topo, hence why its stored in the first as its a pain to access
  return hitInfo;

}


ElectronSeed::PMVars::PMVars():
  dRZPos(std::numeric_limits<float>::infinity()),
  dRZNeg(std::numeric_limits<float>::infinity()),
  dPhiPos(std::numeric_limits<float>::infinity()),
  dPhiNeg(std::numeric_limits<float>::infinity()),
  detId(0),
  layerOrDiskNr(-1)
{}

void ElectronSeed::PMVars::setDPhi(float pos,float neg)
{
  dPhiPos = pos;
  dPhiNeg = neg;
}

void ElectronSeed::PMVars::setDRZ(float pos,float neg)
{
  dRZPos = pos;
  dRZNeg = neg;
}

void ElectronSeed::PMVars::setDet(int iDetId,int iLayerOrDiskNr)
{
  detId = iDetId;
  layerOrDiskNr = iLayerOrDiskNr;
}




