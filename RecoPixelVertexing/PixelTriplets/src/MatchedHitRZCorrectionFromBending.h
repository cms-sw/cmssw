#ifndef MatchedHitRZCorrectionFromBending_H
#define MatchedHitRZCorrectionFromBending_H

#include "DataFormats/DetId/interface/DetId.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class ThirdHitPredictionFromCircle;
class DetLayer;

class MatchedHitRZCorrectionFromBending {
  public:
    MatchedHitRZCorrectionFromBending() : rFixup(0), zFixup(0) {}
    MatchedHitRZCorrectionFromBending(DetId detId);
    MatchedHitRZCorrectionFromBending(const DetLayer *layer);

    inline void operator()(const ThirdHitPredictionFromCircle &pred,
                           double curvature, const TransientTrackingRecHit &hit,
                           double &r, double &z) const
    {
      if (!rFixup && !zFixup) return;
      if (rFixup) r += rFixup(pred, curvature, z, hit);
      if (zFixup) z += zFixup(pred, curvature, r, hit);
    }

  private:
    typedef double (*FixupFn)(const ThirdHitPredictionFromCircle &pred,
                              double curvature, double rOrZ,
                              const TransientTrackingRecHit &hit);

    static double tibMatchedHitZFixup(const ThirdHitPredictionFromCircle &pred,
                                      double curvature, double rOrZ,
                                      const TransientTrackingRecHit &hit);

    FixupFn rFixup, zFixup;
};

#endif // MatchedHitRZCorrectionFromBending_H
