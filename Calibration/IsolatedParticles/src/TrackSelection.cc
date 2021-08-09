#include "Calibration/IsolatedParticles/interface/TrackSelection.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <sstream>

namespace spr {

  bool goodTrack(const reco::Track* pTrack,
                 math::XYZPoint leadPV,
                 spr::trackSelectionParameters parameters,
                 bool debug) {
    bool select = pTrack->quality(parameters.minQuality);
    double dxy = pTrack->dxy(leadPV);
    double dz = pTrack->dz(leadPV);
    double dpbyp = 999;
    if (std::abs(pTrack->qoverp()) > 0.0000001)
      dpbyp = std::abs(pTrack->qoverpError() / pTrack->qoverp());

    if (debug)
      edm::LogVerbatim("IsoTrack") << "Track:: Pt " << pTrack->pt() << " dxy " << dxy << " dz " << dz << " Chi2 "
                                   << pTrack->normalizedChi2() << " dpbyp " << dpbyp << " Quality " << select;

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

      if (debug) {
        edm::LogVerbatim("IsoTrack") << "Default Hit Pattern with "
                                     << hitp.numberOfAllHits(reco::HitPattern::TRACK_HITS) << " hits";
        for (int i = 0; i < hitp.numberOfAllHits(reco::HitPattern::TRACK_HITS); i++) {
          std::ostringstream st1;
          hitp.printHitPattern(reco::HitPattern::TRACK_HITS, i, st1);
          edm::LogVerbatim("IsoTrack") << st1.str();
        }
      }
    }
    if (parameters.maxInMiss >= 0) {
      const reco::HitPattern& hitp = pTrack->hitPattern();
      if (hitp.trackerLayersWithoutMeasurement(reco::HitPattern::MISSING_INNER_HITS) > parameters.maxInMiss)
        select = false;
      if (debug) {
        edm::LogVerbatim("IsoTrack") << "Inner Hit Pattern with "
                                     << hitp.numberOfAllHits(reco::HitPattern::MISSING_INNER_HITS) << " hits";
        for (int i = 0; i < hitp.numberOfAllHits(reco::HitPattern::MISSING_INNER_HITS); i++) {
          std::ostringstream st1;
          hitp.printHitPattern(reco::HitPattern::MISSING_INNER_HITS, i, st1);
          edm::LogVerbatim("IsoTrack") << st1.str();
        }
      }
    }
    if (parameters.maxOutMiss >= 0) {
      const reco::HitPattern& hitp = pTrack->hitPattern();
      if (hitp.trackerLayersWithoutMeasurement(reco::HitPattern::MISSING_OUTER_HITS) > parameters.maxOutMiss)
        select = false;
      if (debug) {
        edm::LogVerbatim("IsoTrack") << "Outer Hit Pattern with "
                                     << hitp.numberOfAllHits(reco::HitPattern::MISSING_OUTER_HITS) << " hits";
        for (int i = 0; i < hitp.numberOfAllHits(reco::HitPattern::MISSING_OUTER_HITS); i++) {
          std::ostringstream st1;
          hitp.printHitPattern(reco::HitPattern::MISSING_OUTER_HITS, i, st1);
          edm::LogVerbatim("IsoTrack") << st1.str();
        }
      }
    }
    if (debug)
      edm::LogVerbatim("IsoTrack") << "Final Selection Result " << select;
    return select;
  }
}  // namespace spr
