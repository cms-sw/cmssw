#ifndef DataFormats_TrackSoA_interface_alpaka_TrackUtilities_h
#define DataFormats_TrackSoA_interface_alpaka_TrackUtilities_h

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

namespace pixelTrack {

  template <typename TrackerTraits, typename Enable = void>
  struct QualityCutsT {};

  template <typename TrackerTraits>
  struct QualityCutsT<TrackerTraits, pixelTopology::isPhase1Topology<TrackerTraits>> {
    using TrackSoAView = reco::TrackSoAView;
    using TrackSoAConstView = reco::TrackSoAConstView;
    float chi2Coeff[4];
    float chi2MaxPt;  // GeV
    float chi2Scale;

    struct Region {
      float maxTip;  // cm
      float minPt;   // GeV
      float maxZip;  // cm
    };

    Region triplet;
    Region quadruplet;

    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool isHP(const TrackSoAConstView &tracks, int nHits, int it) const {
      // impose "region cuts" based on the fit results (phi, Tip, pt, cotan(theta)), Zip)
      // default cuts:
      //   - for triplets:    |Tip| < 0.3 cm, pT > 0.5 GeV, |Zip| < 12.0 cm
      //   - for quadruplets: |Tip| < 0.5 cm, pT > 0.3 GeV, |Zip| < 12.0 cm
      // (see CAHitNtupletGeneratorGPU.cc)
      auto const &region = (nHits > 3) ? quadruplet : triplet;
      return (std::abs(reco::tip(tracks, it)) < region.maxTip) and (tracks.pt(it) > region.minPt) and
             (std::abs(reco::zip(tracks, it)) < region.maxZip);
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool strictCut(const TrackSoAConstView &tracks, int it) const {
      auto roughLog = [](float x) {
        // max diff [0.5,12] at 1.25 0.16143
        // average diff  0.0662998
        union IF {
          uint32_t i;
          float f;
        };
        IF z;
        z.f = x;
        uint32_t lsb = 1 < 21;
        z.i += lsb;
        z.i >>= 21;
        auto f = z.i & 3;
        int ex = int(z.i >> 2) - 127;

        // log2(1+0.25*f)
        // averaged over bins
        const float frac[4] = {0.160497f, 0.452172f, 0.694562f, 0.901964f};
        return float(ex) + frac[f];
      };

      float pt = std::min<float>(tracks.pt(it), chi2MaxPt);
      float chi2Cut = chi2Scale * (chi2Coeff[0] + roughLog(pt) * chi2Coeff[1]);
      if (tracks.chi2(it) >= chi2Cut) {
#ifdef NTUPLE_FIT_DEBUG
        printf("Bad chi2 %d pt %f eta %f chi2 %f\n", it, tracks.pt(it), tracks.eta(it), tracks.chi2(it));
#endif
        return true;
      }
      return false;
    }
  };

  template <typename TrackerTraits>
  struct QualityCutsT<TrackerTraits, pixelTopology::isPhase2Topology<TrackerTraits>> {
    using TrackSoAView = reco::TrackSoAView;
    using TrackSoAConstView = reco::TrackSoAConstView;

    float maxChi2;
    float minPt;
    float maxTip;
    float maxZip;

    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool isHP(const TrackSoAConstView &tracks, int nHits, int it) const {
      return (std::abs(reco::tip(tracks, it)) < maxTip) and (tracks.pt(it) > minPt) and
             (std::abs(reco::zip(tracks, it)) < maxZip);
    }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool strictCut(const TrackSoAConstView &tracks, int it) const {
      return tracks.chi2(it) >= maxChi2;
    }
  };

}  // namespace pixelTrack

#endif  // DataFormats_TrackSoA_interface_alpaka_TrackUtilities_h
