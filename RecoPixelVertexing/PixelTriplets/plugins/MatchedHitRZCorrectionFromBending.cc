#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromCircle.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "MatchedHitRZCorrectionFromBending.h"

// Note: only the TIB layers seem to cause a significant effect,
//       so only z correction is implemented

MatchedHitRZCorrectionFromBending::
    MatchedHitRZCorrectionFromBending(DetId detId)
  : rFixup(0), zFixup(0)
{
  if (detId.subdetId() == SiStripDetId::TIB &&
      TIBDetId(detId).isDoubleSide())
    zFixup = tibMatchedHitZFixup;
}

MatchedHitRZCorrectionFromBending::
    MatchedHitRZCorrectionFromBending(const DetLayer *layer)
  : rFixup(0), zFixup(0)
{
  if (layer->subDetector() == GeomDetEnumerators::TIB) {
    const GeometricSearchDet *tibLayer = layer;
    TIBDetId tibDetId(tibLayer->basicComponents()[0]->geographicalId());
    if (tibDetId.isDoubleSide())
      zFixup = tibMatchedHitZFixup;
  }
}

double MatchedHitRZCorrectionFromBending::
    tibMatchedHitZFixup(const ThirdHitPredictionFromCircle &pred,
                        double curvature, double r,
                        const TransientTrackingRecHit &hit)
{
  // the factors for [ TIB1=0, TIB2=1 ] [ inner string=0, outer string=1 ]
  static const double factors[2][2] = { { -2.4, 2.4 }, { 2.4, -2.4 } };

  TIBDetId det(hit.det()->geographicalId());
  unsigned int layer = det.layer() - 1;
  unsigned int string = !det.isInternalString();
  return factors[layer][string] * pred.angle(curvature, r);
}
