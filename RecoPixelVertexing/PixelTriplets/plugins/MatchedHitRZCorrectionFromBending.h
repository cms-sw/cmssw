#ifndef MatchedHitRZCorrectionFromBending_H
#define MatchedHitRZCorrectionFromBending_H

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

class ThirdHitPredictionFromCircle;
class DetLayer;
class TrackerTopology;

class MatchedHitRZCorrectionFromBending {
public:
  MatchedHitRZCorrectionFromBending() : rFixup(nullptr), zFixup(nullptr) {}
  MatchedHitRZCorrectionFromBending(DetId detId, const TrackerTopology *tTopo);
  MatchedHitRZCorrectionFromBending(const DetLayer *layer, const TrackerTopology *tTopo);

  inline void operator()(const ThirdHitPredictionFromCircle &pred,
                         double curvature,
                         const TrackingRecHit &hit,
                         double &r,
                         double &z,
                         const TrackerTopology *tTopo) const {
    if (!rFixup && !zFixup)
      return;
    if (rFixup)
      r += rFixup(pred, curvature, z, hit, tTopo);
    if (zFixup)
      z += zFixup(pred, curvature, r, hit, tTopo);
  }

private:
  typedef double (*FixupFn)(const ThirdHitPredictionFromCircle &pred,
                            double curvature,
                            double rOrZ,
                            const TrackingRecHit &hit,
                            const TrackerTopology *tTopo);

  static double tibMatchedHitZFixup(const ThirdHitPredictionFromCircle &pred,
                                    double curvature,
                                    double rOrZ,
                                    const TrackingRecHit &hit,
                                    const TrackerTopology *tTopo);

  FixupFn rFixup, zFixup;
};

#endif  // MatchedHitRZCorrectionFromBending_H
