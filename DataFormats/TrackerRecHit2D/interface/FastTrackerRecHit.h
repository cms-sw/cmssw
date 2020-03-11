/**
 * base class for FastSim tracker RecHit containers
 * it inherits from BaseTrackerRecHit
 * and adds all the special FastSim features required to 
 * - perform truth matching,
 * - duplicate track removal
 * - fast tracking emulation
 */

#ifndef FastTrackerRecHit_H
#define FastTrackerRecHit_H

#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"

namespace fastTrackerRecHitType {
  enum HitType {
    siPixel = 0,
    siStrip1D = 1,
    siStrip2D = 2,
    siStripMatched2D = 3,
    siStripProjectedMono2D = 4,
    siStripProjectedStereo2D = 5,
  };
  inline trackerHitRTTI::RTTI rtti(HitType hitType) {
    if (hitType >= siPixel && hitType <= siStrip2D)
      return trackerHitRTTI::fastSingle;
    else if (hitType == siStripMatched2D)
      return trackerHitRTTI::fastMatch;
    else if (hitType == siStripProjectedMono2D)
      return trackerHitRTTI::fastProjMono;
    else if (hitType == siStripProjectedStereo2D)
      return trackerHitRTTI::fastProjStereo;
    else
      return trackerHitRTTI::undef;
  }
  inline bool is2D(HitType hitType) { return hitType != siStrip1D; }
  inline bool isPixel(HitType hitType) { return hitType == siPixel; }
}  // namespace fastTrackerRecHitType

class FastTrackerRecHit : public BaseTrackerRecHit {
public:
  /// default constructor
  ///
  FastTrackerRecHit() : BaseTrackerRecHit(), isPixel_(false), is2D_(true), recHitCombinationIndex_(-1) {}

  /// destructor
  ///
  ~FastTrackerRecHit() override {}

  /// constructor
  /// requires a position with error in local detector coordinates,
  /// the detector id, and type information (rt)
  FastTrackerRecHit(const LocalPoint& p,
                    const LocalError& e,
                    GeomDet const& idet,
                    fastTrackerRecHitType::HitType hitType)
      : BaseTrackerRecHit(p, e, idet, fastTrackerRecHitType::rtti(hitType)),
        isPixel_(fastTrackerRecHitType::isPixel(hitType)),
        is2D_(fastTrackerRecHitType::is2D(hitType)),
        recHitCombinationIndex_(-1),
        energyLoss_(0.0)  //holy golden seal
  {
    store();
  }

  FastTrackerRecHit* clone() const override {
    FastTrackerRecHit* p = new FastTrackerRecHit(*this);
    p->load();
    return p;
  }

  /// Steers behaviour of hit in track fit.
  /// Hit is interpreted as 1D or 2D depending on value of is2D_

  float energyLoss() const { return energyLoss_; }  // holy golden seal
  void setEnergyLoss(float e) { energyLoss_ = e; }  //   holy golden seal was a virtual void...
  void getKfComponents(KfComponentsHolder& holder) const override {
    if (is2D_)
      getKfComponents2D(holder);
    else
      getKfComponents1D(holder);
  }

  /// Steers behaviour of hit in track fit.
  /// Hit is interpreted as 1D or 2D depending on value of is2D_
  int dimension() const override { return is2D_ ? 2 : 1; }  ///< get the dimensions right

  /// Steers behaviour of hit in track fit.
  /// FastSim hit smearing assumes
  bool canImproveWithTrack() const override { return false; }

  /* getters */

  virtual size_t nIds() const { return 0; }
  virtual int32_t id(size_t i = 0) const { return -1; }
  virtual int32_t eventId(size_t i = 0) const { return -1; }

  virtual size_t nSimTrackIds() const { return 0; }
  virtual int32_t simTrackId(size_t i) const { return -1; }
  virtual int32_t simTrackEventId(size_t i) const { return -1; }

  virtual int32_t recHitCombinationIndex() const { return recHitCombinationIndex_; }

  bool isPixel() const override { return isPixel_; }  ///< pixel or strip?

  /* setters */

  virtual void setEventId(int32_t eventId){};

  void set2D(bool is2D = true) { is2D_ = is2D; }

  virtual void setRecHitCombinationIndex(int32_t recHitCombinationIndex) {
    recHitCombinationIndex_ = recHitCombinationIndex;
  }

  /// bogus function :
  /// implement purely virtual function of TrackingRecHit
  std::vector<const TrackingRecHit*> recHits() const override { return std::vector<TrackingRecHit const*>(); }

  /// bogus function :
  /// implement purely virtual function of TrackingRecHit
  std::vector<TrackingRecHit*> recHits() override { return std::vector<TrackingRecHit*>(); }

  /// bogus function :
  /// implement purely virutal function of BaseTrackerRecHit
  OmniClusterRef const& firstClusterRef() const override;

  /// fastsim's way to check whether 2 single hits share sim-information or not
  /// hits are considered to share sim-information if
  /// - they have the same hit id number
  /// - they have the same event id number
  // used by functions
  // - FastTrackerSingleRecHit::sharesInput
  // - FastSiStripMatchedRecHit::sharesInput
  // - FastProjectedSiStripRecHit2D::sharesInput
  inline bool sameId(const FastTrackerRecHit* other, size_t i = 0, size_t j = 0) const {
    return id(i) == other->id(j) && eventId(i) == other->eventId(j);
  }
  inline bool sharesInput(const TrackingRecHit* other, SharedInputType what) const override {
    // cast other hit
    if (!trackerHitRTTI::isFast(*other))
      return false;
    const FastTrackerRecHit* otherCast = static_cast<const FastTrackerRecHit*>(other);

    // checks
    if (this->nIds() == 1) {         // this hit is single/projected
      if (otherCast->nIds() == 1) {  // other hit is single/projected
        return this->sameId(otherCast, 0, 0);
      } else {  // other hit is matched
        if (what == all) {
          return false;
        } else {
          return (this->sameId(otherCast, 0, 0) || this->sameId(otherCast, 0, 1));
        }
      }
    } else {                         // this hit is matched
      if (otherCast->nIds() == 1) {  // other hit is single/projected
        if (what == all) {
          return false;
        } else {
          return (this->sameId(otherCast, 0, 0) || this->sameId(otherCast, 1, 0));
        }
      } else {  // other hit is matched
        if (what == all) {
          return (this->sameId(otherCast, 0, 0) && this->sameId(otherCast, 1, 1));
        } else {
          return (this->sameId(otherCast, 0, 0) || this->sameId(otherCast, 1, 1));
        }
      }
    }
  }

protected:
  const bool isPixel_;  ///< hit is either on pixel modul (isPixel_ = true) or strip module (isPixel_ = false)
  bool is2D_;           ///< hit is either one dimensional (is2D_ = false) or two dimensions (is2D_ = true)

  LocalPoint myPos_;  ///< helps making the hit postion and error persistent
  LocalError myErr_;  ///< helps making the hit postion and error persistent

  void store() {
    myPos_ = pos_;
    myErr_ = err_;
  }  ///< helps making the hit postion and error persistent
  void load() {
    pos_ = myPos_;
    err_ = myErr_;
  }  ///< helps making the hit postion and error persistent

  uint32_t recHitCombinationIndex_;
  float energyLoss_;  //holy golden seal

protected:
  FastTrackerRecHit* clone_(TkCloner const& cloner, TrajectoryStateOnSurface const& tsos) const override {
    return this->clone();
  }
};

/// Comparison operator
///
inline bool operator<(const FastTrackerRecHit& one, const FastTrackerRecHit& other) {
  return (one.geographicalId() < other.geographicalId());
}

#endif
