#ifndef DataFormats_EgammaReco_ElectronSeed_h
#define DataFormats_EgammaReco_ElectronSeed_h

//********************************************************************
//
// A verson of reco::ElectronSeed which can have N hits as part of the
// 2017 upgrade of E/gamma pixel matching for the phaseI pixels
//
// author: S. Harper (RAL), 2017
//
//notes:
// While it is technically named ElectronSeed, it is effectively a new class
// However to simplify things, the name ElectronSeed was kept
// (trust me it was simplier...)
//
// Noticed that h/e values never seem to used anywhere and they are a
// mild pain to propagate in the new framework so they were removed
//
// infinities are used to mark invalid unset values to maintain
// compatibilty with the orginal ElectronSeed class
//
//description:
// An ElectronSeed is a TrajectorySeed with E/gamma specific information
// A TrajectorySeed has a series of hits associated with it
// (accessed by TrajectorySeed::nHits(), TrajectorySeed::recHits())
// and ElectronSeed stores which of those hits match well to a supercluster
// together with the matching parameters (this is known as EcalDriven).
// ElectronSeeds can be TrackerDriven in which case the matching is not done.
// It used to be fixed to two matched hits, now this is an arbitary number
// Its designed with pixel matching with mind but tries to be generally
// applicable to strips as well.
// It is worth noting that due to different ways ElectronSeeds can be created
// they do not always have all parameters filled
//
//********************************************************************

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

namespace reco {

  class ElectronSeed : public TrajectorySeed {
  public:
    struct PMVars {
      float dRZPos;
      float dRZNeg;
      float dPhiPos;
      float dPhiNeg;
      int detId;          //this is already stored as the hit is stored in traj seed but a useful sanity check
      int layerOrDiskNr;  //redundant as stored in detId but its a huge pain to hence why its saved here

      PMVars();

      void setDPhi(float pos, float neg);
      void setDRZ(float pos, float neg);
      void setDet(int iDetId, int iLayerOrDiskNr);
    };

    typedef edm::OwnVector<TrackingRecHit> RecHitContainer;
    typedef edm::RefToBase<CaloCluster> CaloClusterRef;
    typedef edm::Ref<TrackCollection> CtfTrackRef;

    static std::string const& name() {
      static std::string const name_("ElectronSeed");
      return name_;
    }

    //! Construction of base attributes
    ElectronSeed();
    ElectronSeed(const TrajectorySeed&);
    ElectronSeed(PTrajectoryStateOnDet& pts, RecHitContainer& rh, PropagationDirection& dir);
    ElectronSeed* clone() const override { return new ElectronSeed(*this); }
    ~ElectronSeed() override;

    //! Set additional info
    void setCtfTrack(const CtfTrackRef&);
    void setCaloCluster(const CaloClusterRef& clus) {
      caloCluster_ = clus;
      isEcalDriven_ = true;
    }
    void addHitInfo(const PMVars& hitVars) { hitInfo_.push_back(hitVars); }
    void setNrLayersAlongTraj(int val) { nrLayersAlongTraj_ = val; }
    //! Accessors
    const CtfTrackRef& ctfTrack() const { return ctfTrack_; }
    const CaloClusterRef& caloCluster() const { return caloCluster_; }

    //! Utility
    TrackCharge getCharge() const { return startingState().parameters().charge(); }

    bool isEcalDriven() const { return isEcalDriven_; }
    bool isTrackerDriven() const { return isTrackerDriven_; }

    const std::vector<PMVars>& hitInfo() const { return hitInfo_; }
    float dPhiNeg(size_t hitNr) const { return getVal(hitNr, &PMVars::dPhiNeg); }
    float dPhiPos(size_t hitNr) const { return getVal(hitNr, &PMVars::dPhiPos); }
    float dPhiBest(size_t hitNr) const { return bestVal(dPhiNeg(hitNr), dPhiPos(hitNr)); }
    float dRZPos(size_t hitNr) const { return getVal(hitNr, &PMVars::dRZPos); }
    float dRZNeg(size_t hitNr) const { return getVal(hitNr, &PMVars::dRZNeg); }
    float dRZBest(size_t hitNr) const { return bestVal(dRZNeg(hitNr), dRZPos(hitNr)); }
    int detId(size_t hitNr) const { return hitNr < hitInfo_.size() ? hitInfo_[hitNr].detId : 0; }
    int subDet(size_t hitNr) const { return DetId(detId(hitNr)).subdetId(); }
    int layerOrDiskNr(size_t hitNr) const { return getVal(hitNr, &PMVars::layerOrDiskNr); }
    int nrLayersAlongTraj() const { return nrLayersAlongTraj_; }

    unsigned int hitsMask() const;
    void initTwoHitSeed(const unsigned char hitMask);
    void setNegAttributes(const float dRZ2 = std::numeric_limits<float>::infinity(),
                          const float dPhi2 = std::numeric_limits<float>::infinity(),
                          const float dRZ1 = std::numeric_limits<float>::infinity(),
                          const float dPhi1 = std::numeric_limits<float>::infinity());
    void setPosAttributes(const float dRZ2 = std::numeric_limits<float>::infinity(),
                          const float dPhi2 = std::numeric_limits<float>::infinity(),
                          const float dRZ1 = std::numeric_limits<float>::infinity(),
                          const float dPhi1 = std::numeric_limits<float>::infinity());

    //this is a backwards compatible function designed to
    //convert old format ElectronSeeds to the new format
    //only public due to root io rules, not intended for any other use
    //also in theory not necessary to part of this class
    static std::vector<PMVars> createHitInfo(const float dPhi1Pos,
                                             const float dPhi1Neg,
                                             const float dRZ1Pos,
                                             const float dRZ1Neg,
                                             const float dPhi2Pos,
                                             const float dPhi2Neg,
                                             const float dRZ2Pos,
                                             const float dRZ2Neg,
                                             const char hitMask,
                                             const TrajectorySeed::range recHits);

  private:
    static float bestVal(float val1, float val2) { return std::abs(val1) < std::abs(val2) ? val1 : val2; }
    template <typename T>
    T getVal(unsigned int hitNr, T PMVars::*val) const {
      return hitNr < hitInfo_.size() ? hitInfo_[hitNr].*val : std::numeric_limits<T>::infinity();
    }
    static std::vector<unsigned int> hitNrsFromMask(unsigned int hitMask);

  private:
    CtfTrackRef ctfTrack_;
    CaloClusterRef caloCluster_;
    std::vector<PMVars> hitInfo_;
    int nrLayersAlongTraj_;

    bool isEcalDriven_;
    bool isTrackerDriven_;
  };
}  // namespace reco

#endif
