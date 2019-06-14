#ifndef PreId_H
#define PreId_H

// Author F. Beaudette. March 2010

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

namespace reco {
  class PreId {
  public:
    enum MatchingType { NONE = 0, ECALMATCH = 1, ESMATCH = 2, TRACKFILTERING = 3, MVA = 4, FINAL = 10 };

  public:
    PreId(unsigned nselection = 1)
        : trackRef_(reco::TrackRef()),
          clusterRef_(reco::PFClusterRef()),
          matchingEop_(-999.),
          EcalPos_(math::XYZPoint()),
          meanShower_(math::XYZPoint()),
          gsfChi2_(-999.),
          dpt_(0.),
          chi2Ratio_(0.) {
      matching_.resize(nselection, false);
      mva_.resize(nselection, -999.);
      geomMatching_.resize(5, -999.);
    }
    void setTrack(reco::TrackRef trackref) { trackRef_ = trackref; }

    void setECALMatchingProperties(PFClusterRef clusterRef,
                                   const math::XYZPoint &ecalpos,
                                   const math::XYZPoint &meanShower,
                                   float deta,
                                   float dphi,
                                   float chieta,
                                   float chiphi,
                                   float chi2,
                                   float eop) {
      clusterRef_ = clusterRef;
      EcalPos_ = ecalpos;
      meanShower_ = meanShower;
      geomMatching_[0] = deta;
      geomMatching_[1] = dphi;
      geomMatching_[2] = chieta;
      geomMatching_[3] = chiphi;
      geomMatching_[4] = chi2;
      matchingEop_ = eop;
    }

    void setTrackProperties(float newchi2, float chi2ratio, float dpt) {
      gsfChi2_ = newchi2;
      chi2Ratio_ = chi2ratio;
      dpt_ = dpt;
    }

    void setFinalDecision(bool accepted, unsigned n = 0) { setMatching(FINAL, accepted, n); }
    void setECALMatching(bool accepted, unsigned n = 0) { setMatching(ECALMATCH, accepted, n); }
    void setESMatching(bool accepted, unsigned n = 0) { setMatching(ESMATCH, accepted, n); }
    void setTrackFiltering(bool accepted, unsigned n = 0) { setMatching(TRACKFILTERING, accepted, n); }
    void setMVA(bool accepted, float mva, unsigned n = 0) {
      setMatching(MVA, accepted, n);
      if (n < mva_.size())
        mva_[n] = mva;
    }

    void setMatching(MatchingType type, bool result, unsigned n = 0);
    bool matching(MatchingType type, unsigned n = 0) const {
      if (n < matching_.size()) {
        return matching_[n] & (1 << type);
      }
      return false;
    }

    /// Access methods
    inline const std::vector<float> &geomMatching() const { return geomMatching_; }
    inline float eopMatch() const { return matchingEop_; }
    inline float pt() const { return trackRef_->pt(); }
    inline float eta() const { return trackRef_->eta(); }
    inline float kfChi2() const { return trackRef_->normalizedChi2(); }
    inline float kfNHits() const { return trackRef_->found(); }

    const math::XYZPoint &ecalPos() const { return EcalPos_; }
    const math::XYZPoint &meanShower() const { return meanShower_; }

    inline float chi2Ratio() const { return chi2Ratio_; }
    inline float gsfChi2() const { return gsfChi2_; }

    inline bool ecalMatching(unsigned n = 0) const { return matching(ECALMATCH, n); }
    inline bool esMatching(unsigned n = 0) const { return matching(ESMATCH, n); }
    inline bool trackFiltered(unsigned n = 0) const { return matching(TRACKFILTERING, n); }
    inline bool mvaSelected(unsigned n = 0) const { return matching(MVA, n); }
    inline bool preIded(unsigned n = 0) const { return matching(FINAL, n); }

    float mva(unsigned n = 0) const;
    inline float dpt() const { return dpt_; }
    reco::TrackRef trackRef() const { return trackRef_; }
    PFClusterRef clusterRef() const { return clusterRef_; }

  private:
    reco::TrackRef trackRef_;
    PFClusterRef clusterRef_;

    std::vector<float> geomMatching_;
    float matchingEop_;
    math::XYZPoint EcalPos_;
    math::XYZPoint meanShower_;

    float gsfChi2_;
    float dpt_;
    float chi2Ratio_;
    std::vector<float> mva_;

    //    bool goodpreid_;
    //    bool TkId_;
    //    bool EcalMatching_;
    //    bool PSMatching_;
    std::vector<int> matching_;
  };
}  // namespace reco
#endif
