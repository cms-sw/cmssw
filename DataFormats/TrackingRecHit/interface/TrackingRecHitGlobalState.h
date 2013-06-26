#ifndef TrackingRecHitGlobalState_H
#define TrackingRecHitGlobalState_H


#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

// position and error in global coord
struct TrackingRecHitGlobalState {
  using Vector = Basic3DVector<float>;

  Vector position;
  float r,phi;
  float errorR,errorZ,errorRPhi;

};


#endif
