#ifndef MuonIdentification_MuonArbitrationMethods_h
#define MuonIdentification_MuonArbitrationMethods_h

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonReco/interface/MuonChamberMatch.h"

#include <utility>

// Author: Jake Ribnik (UCSB)

/// functor predicate for standard library sort algorithm
struct SortMuonSegmentMatches {
  /// constructor takes arbitration type
  SortMuonSegmentMatches(unsigned int flag, double dx_norm = 0.45, double dDphiDz_norm = 0.00003) {
    flag_ = flag;
    dx_norm_ = dx_norm;
    dDphiDz_norm_ = dDphiDz_norm;
  }
  /// sorts vector of pairs of chamber and segment pointers
  bool operator()(std::pair<reco::MuonChamberMatch*, reco::MuonSegmentMatch*> p1,
                  std::pair<reco::MuonChamberMatch*, reco::MuonSegmentMatch*> p2) {
    reco::MuonChamberMatch* cm1 = p1.first;
    reco::MuonSegmentMatch* sm1 = p1.second;
    reco::MuonChamberMatch* cm2 = p2.first;
    reco::MuonSegmentMatch* sm2 = p2.second;

    if (flag_ == reco::MuonSegmentMatch::BestInChamberByDX || flag_ == reco::MuonSegmentMatch::BestInStationByDX ||
        flag_ == reco::MuonSegmentMatch::BelongsToTrackByDX)
      return fabs(sm1->x - cm1->x) < fabs(sm2->x - cm2->x);
    if (flag_ == reco::MuonSegmentMatch::BestInChamberByDR || flag_ == reco::MuonSegmentMatch::BestInStationByDR ||
        flag_ == reco::MuonSegmentMatch::BelongsToTrackByDR) {
      if ((!sm1->hasZed()) || (!sm2->hasZed()))  // no y information so return dx
        return fabs(sm1->x - cm1->x) < fabs(sm2->x - cm2->x);
      return sqrt(pow(sm1->x - cm1->x, 2) + pow(sm1->y - cm1->y, 2)) <
             sqrt(pow(sm2->x - cm2->x, 2) + pow(sm2->y - cm2->y, 2));
    }
    if (flag_ == reco::MuonSegmentMatch::BestInChamberByDXSlope ||
        flag_ == reco::MuonSegmentMatch::BestInStationByDXSlope ||
        flag_ == reco::MuonSegmentMatch::BelongsToTrackByDXSlope)
      return fabs(sm1->dXdZ - cm1->dXdZ) < fabs(sm2->dXdZ - cm2->dXdZ);
    if (flag_ == reco::MuonSegmentMatch::BestInChamberByDRSlope ||
        flag_ == reco::MuonSegmentMatch::BestInStationByDRSlope ||
        flag_ == reco::MuonSegmentMatch::BelongsToTrackByDRSlope) {
      if ((!sm1->hasZed()) || (!sm2->hasZed()))  // no y information so return dx
        return fabs(sm1->dXdZ - cm1->dXdZ) < fabs(sm2->dXdZ - cm2->dXdZ);
      return sqrt(pow(sm1->dXdZ - cm1->dXdZ, 2) + pow(sm1->dYdZ - cm1->dYdZ, 2)) <
             sqrt(pow(sm2->dXdZ - cm2->dXdZ, 2) + pow(sm2->dYdZ - cm2->dYdZ, 2));
    }
    if (flag_ == reco::MuonSegmentMatch::BestInChamberByDX_DPhiDZ ||
        flag_ == reco::MuonSegmentMatch::BestInStationByDX_DPhiDZ ||
        flag_ == reco::MuonSegmentMatch::BelongsToTrackByDX_DPhiDZ) {
      if (fabs(sm1->y - cm1->y) > 3 * sqrt(pow(sm1->yErr, 2) + pow(cm1->yErr, 2))) {
        // Bad segment: Dy too large
        return false;
      }
      double dx1 = sm1->x - cm1->x;
      double dDphiDz1 = sm1->dPhidZ - cm1->dPhidZ;
      double dx2 = sm2->x - cm2->x;
      double dDphiDz2 = sm2->dPhidZ - cm2->dPhidZ;

      // normalization factors to make dx and dDPhidZ comparable
      // obtained from the distribution in the noPU scenario
      double dx_norm = dx_norm_;
      double dDphiDz_norm = dDphiDz_norm_;

      double pull_x1 = std::abs(dx1 / dx_norm);
      double pull_dDphiDz1 = std::abs(dDphiDz1 / dDphiDz_norm);
      double pull_x2 = std::abs(dx2 / dx_norm);
      double pull_dDphiDz2 = std::abs(dDphiDz2 / dDphiDz_norm);

      double D1 = pull_x1 * pull_x1 + pull_dDphiDz1 * pull_dDphiDz1;
      double D2 = pull_x2 * pull_x2 + pull_dDphiDz2 * pull_dDphiDz2;

      return D1 < D2;
    }

    return false;  // is this appropriate? fix this
  }

  unsigned int flag_;
  double dx_norm_;
  double dDphiDz_norm_;
};

#endif
