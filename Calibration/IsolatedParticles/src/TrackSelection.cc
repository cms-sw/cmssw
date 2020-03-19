#include "Calibration/IsolatedParticles/interface/TrackSelection.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"

#include <iostream>

//#define EDM_ML_DEBUG

namespace spr {

  bool goodTrack(const reco::Track* pTrack,
                 math::XYZPoint leadPV,
                 spr::trackSelectionParameters parameters,
                 bool
#ifdef EDM_ML_DEBUG
                     debug
#endif
  ) {

    bool select = pTrack->quality(parameters.minQuality);
    double dxy = pTrack->dxy(leadPV);
    double dz = pTrack->dz(leadPV);
    double dpbyp = 999;
    if (std::abs(pTrack->qoverp()) > 0.0000001)
      dpbyp = std::abs(pTrack->qoverpError() / pTrack->qoverp());

#ifdef EDM_ML_DEBUG
    if (debug)
      std::cout << "Track:: Pt " << pTrack->pt() << " dxy " << dxy << " dz " << dz << " Chi2 "
                << pTrack->normalizedChi2() << " dpbyp " << dpbyp << " Quality " << select << std::endl;
#endif
    if (pTrack->pt() < parameters.minPt)
      select = false;
    if (dxy > parameters.maxDxyPV || dz > parameters.maxDzPV)
      select = false;
    if (pTrack->normalizedChi2() > parameters.maxChi2)
      select = false;
    if (dpbyp > parameters.maxDpOverP)
      select = false;

    if (parameters.minLayerCrossed > 0 || parameters.minOuterHit > 0) {
      const reco::HitPattern& hitp = pTrack->hitPattern();
      if (parameters.minLayerCrossed > 0 && hitp.trackerLayersWithMeasurement() < parameters.minLayerCrossed)
        select = false;
      if (parameters.minOuterHit > 0 &&
          (hitp.stripTOBLayersWithMeasurement() + hitp.stripTECLayersWithMeasurement()) < parameters.minOuterHit)
        select = false;
#ifdef EDM_ML_DEBUG
      if (debug) {
        std::cout << "Default Hit Pattern with " << hitp.numberOfAllHits(reco::HitPattern::TRACK_HITS) << " hits"
                  << std::endl;
        for (int i = 0; i < hitp.numberOfAllHits(reco::HitPattern::TRACK_HITS); i++)
          hitp.printHitPattern(reco::HitPattern::TRACK_HITS, i, std::cout);
      }
#endif
    }
    if (parameters.maxInMiss >= 0) {
      const reco::HitPattern& hitp = pTrack->hitPattern();
      if (hitp.trackerLayersWithoutMeasurement(reco::HitPattern::MISSING_INNER_HITS) > parameters.maxInMiss)
        select = false;
#ifdef EDM_ML_DEBUG
      if (debug) {
        std::cout << "Inner Hit Pattern with " << hitp.numberOfAllHits(reco::HitPattern::MISSING_INNER_HITS) << " hits"
                  << std::endl;
        for (int i = 0; i < hitp.numberOfAllHits(reco::HitPattern::MISSING_INNER_HITS); i++)
          hitp.printHitPattern(reco::HitPattern::MISSING_INNER_HITS, i, std::cout);
      }
#endif
    }
    if (parameters.maxOutMiss >= 0) {
      const reco::HitPattern& hitp = pTrack->hitPattern();
      if (hitp.trackerLayersWithoutMeasurement(reco::HitPattern::MISSING_OUTER_HITS) > parameters.maxOutMiss)
        select = false;
#ifdef EDM_ML_DEBUG
      if (debug) {
        std::cout << "Outer Hit Pattern with " << hitp.numberOfAllHits(reco::HitPattern::MISSING_OUTER_HITS) << " hits"
                  << std::endl;
        for (int i = 0; i < hitp.numberOfAllHits(reco::HitPattern::MISSING_OUTER_HITS); i++)
          hitp.printHitPattern(reco::HitPattern::MISSING_OUTER_HITS, i, std::cout);
      }
#endif
    }
#ifdef EDM_ML_DEBUG
    if (debug)
      std::cout << "Final Selection Result " << select << std::endl;
#endif
    return select;
  }
}  // namespace spr
