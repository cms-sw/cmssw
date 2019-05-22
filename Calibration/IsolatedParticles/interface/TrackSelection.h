#ifndef CalibrationIsolatedParticlesTrackSelection_h
#define CalibrationIsolatedParticlesTrackSelection_h

// system include files
#include <cmath>

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/TrackReco/interface/Track.h"

namespace spr {

  struct trackSelectionParameters {
    trackSelectionParameters() {
      minPt = 0;
      minQuality = reco::TrackBase::highPurity;
      maxDxyPV = maxDzPV = 999999.;
      maxChi2 = maxDpOverP = 99999999., minOuterHit = minLayerCrossed = 0;
      maxInMiss = maxOutMiss = -1;
    }
    double minPt;
    reco::TrackBase::TrackQuality minQuality;
    double maxDxyPV, maxDzPV, maxChi2, maxDpOverP;
    int minOuterHit, minLayerCrossed;
    int maxInMiss, maxOutMiss;
  };

  bool goodTrack(const reco::Track* pTrack,
                 math::XYZPoint leadPV,
                 trackSelectionParameters parameters,
                 bool debug = false);

}  // namespace spr
#endif
