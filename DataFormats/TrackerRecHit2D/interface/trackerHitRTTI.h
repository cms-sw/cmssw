#ifndef DataFormats_TrackerRecHit2D_trackerHitRTTI_H
#define DataFormats_TrackerRecHit2D_trackerHitRTTI_H

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

namespace trackerHitRTTI {
  // tracking hit can be : single (si1D, si2D, pix), projected, matched or multi
  enum RTTI {
    undef = 0,
    single = 1,
    projStereo = 2,
    projMono = 3,
    match = 4,
    multi = 5,
    fastSingle = 6,
    fastProjStereo = 7,
    fastProjMono = 8,
    fastMatch = 9,
    notFromCluster = 10,
    mipTiming = 11,
    vector = 12
  };
  inline RTTI rtti(TrackingRecHit const& hit) { return RTTI(hit.getRTTI()); }
  inline bool isUndef(TrackingRecHit const& hit) { return rtti(hit) == undef; }
  inline bool isNotFromCluster(TrackingRecHit const& hit) { return rtti(hit) == notFromCluster; }
  inline bool isSingle(TrackingRecHit const& hit) { return rtti(hit) == single || rtti(hit) == fastSingle; }
  inline bool isProjMono(TrackingRecHit const& hit) { return rtti(hit) == projMono || rtti(hit) == fastProjMono; }
  inline bool isProjStereo(TrackingRecHit const& hit) { return rtti(hit) == projStereo || rtti(hit) == fastProjStereo; }
  inline bool isProjected(TrackingRecHit const& hit) {
    return ((rtti(hit) == projMono) || (rtti(hit) == projStereo)) || (rtti(hit) == fastProjMono) ||
           (rtti(hit) == fastProjStereo);
  }
  inline bool isMatched(TrackingRecHit const& hit) { return rtti(hit) == match || rtti(hit) == fastMatch; }
  inline bool isMulti(TrackingRecHit const& hit) { return rtti(hit) == multi; }
  inline bool isSingleType(TrackingRecHit const& hit) { return (rtti(hit) > 0) && (rtti(hit) < 4); }
  inline bool isFromDet(TrackingRecHit const& hit) {
    return (((rtti(hit) > 0) && (rtti(hit) < 6)) || (rtti(hit) == 12));
  }
  inline bool isFast(TrackingRecHit const& hit) { return (rtti(hit) > 5) && (rtti(hit) <= 9); }
  inline bool isFromDetOrFast(TrackingRecHit const& hit) {
    return (((rtti(hit) > 0) && (rtti(hit) < 10)) || (rtti(hit) == 12));
  }
  inline bool isTiming(TrackingRecHit const& hit) { return rtti(hit) == mipTiming; }
  inline bool isVector(TrackingRecHit const& hit) { return rtti(hit) == vector; }
  inline unsigned int projId(TrackingRecHit const& hit) { return hit.rawId() + int(rtti(hit)) - 1; }
}  // namespace trackerHitRTTI

#endif
