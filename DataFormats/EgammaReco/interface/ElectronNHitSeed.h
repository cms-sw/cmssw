#ifndef DataFormats_EgammaReco_ElectronNHitSeed_h
#define DataFormats_EgammaReco_ElectronNHitSeed_h

//********************************************************************
//
// A verson of reco::ElectronSeed which can have N hits 
//
// Noticed that h/e values never seem to used anywhere and they are a 
// mild pain to propagate in the new framework so they were removed
// Likewise, the hit mask (we always use all the hits in the seed)
//
// author: S. Harper (RAL), 2017
//
//*********************************************************************


#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectoryState/interface/TrackCharge.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/Ref.h"

#include <vector>
#include <limits>

namespace reco
{
  
  class ElectronNHitSeed : public TrajectorySeed {
  public :
    struct PMVars {
      float dRZPos;
      float dRZNeg;
      float dPhiPos;
      float dPhiNeg;
      int detId; //this is already stored as the hit is stored in traj seed but a useful sanity check
      int layerOrDisk;//redundant as stored in detId but its a huge pain to hence why its saved here

      void setDPhi(float pos,float neg){dPhiPos=pos;dPhiNeg=neg;}
      void setDRZ(float pos,float neg){dRZPos=pos;dRZNeg=neg;}
      void setDet(int iDetId,int iLayerOrDisk){detId=iDetId;layerOrDisk=iLayerOrDisk;}

    };
    
    
    typedef edm::OwnVector<TrackingRecHit> RecHitContainer ;
    typedef edm::RefToBase<CaloCluster> CaloClusterRef ;
    typedef edm::Ref<TrackCollection> CtfTrackRef ;
    static std::string const & name()
    {
      static std::string const name_("ElectronNHitSeed") ;
      return name_;
    }
    
    //! Construction of base attributes
    ElectronNHitSeed() ;
    ElectronNHitSeed( const TrajectorySeed & ) ;
    ElectronNHitSeed( PTrajectoryStateOnDet & pts, RecHitContainer & rh,  PropagationDirection & dir ) ;
    ElectronNHitSeed * clone() const { return new ElectronNHitSeed(*this) ; }
    virtual ~ElectronNHitSeed()=default;

    //! Set additional info
    void setCtfTrack( const CtfTrackRef & ) ;
    void setCaloCluster( const CaloClusterRef& clus){caloCluster_=clus;isEcalDriven_=true;}
    void addHitInfo(const PMVars& hitVars){hitInfo_.push_back(hitVars);}
    void setNrLayersAlongTraj(int val){nrLayersAlongTraj_=val;}
    //! Accessors
    const CtfTrackRef& ctfTrack() const { return ctfTrack_ ; }
    const CaloClusterRef& caloCluster() const { return caloCluster_ ; }
   
    //! Utility
    TrackCharge getCharge() const { return startingState().parameters().charge() ; }

    bool isEcalDriven() const { return isEcalDriven_ ; }
    bool isTrackerDriven() const { return isTrackerDriven_ ; }

    const std::vector<PMVars>& hitInfo()const{return hitInfo_;}
    float dPhiNeg(size_t hitNr)const{return hitInfo_[hitNr].dPhiNeg;}
    float dPhiPos(size_t hitNr)const{return hitInfo_[hitNr].dPhiPos;}
    float dPhiBest(size_t hitNr)const{return bestVal(dPhiNeg(hitNr),dPhiPos(hitNr));}
    float dRZPos(size_t hitNr)const{return hitInfo_[hitNr].dRZPos;}
    float dRZNeg(size_t hitNr)const{return hitInfo_[hitNr].dRZNeg;}
    float dRZBest(size_t hitNr)const{return bestVal(dRZNeg(hitNr),dRZPos(hitNr));}
    int subDet(size_t hitNr)const{return DetId(hitInfo_[hitNr].detId).subdetId();}
    int layerOrDisk(size_t hitNr)const{return hitInfo_[hitNr].layerOrDisk;}
    int nrLayersAlongTraj()const{return nrLayersAlongTraj_;}
    

  private:
    static float bestVal(float val1,float val2){return std::abs(val1)<std::abs(val2) ? val1 : val2;}
    
  private:

    CtfTrackRef ctfTrack_ ;
    CaloClusterRef caloCluster_ ;
    std::vector<PMVars> hitInfo_;
    int nrLayersAlongTraj_;
    
    bool isEcalDriven_ ;
    bool isTrackerDriven_ ;

  };
}

#endif
