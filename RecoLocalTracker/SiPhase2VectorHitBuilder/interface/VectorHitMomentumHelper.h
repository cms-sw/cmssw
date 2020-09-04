#ifndef VectorHitMomentumHelper_H
#define VectorHitMomentumHelper_H

#include "DataFormats/TrackerRecHit2D/interface/VectorHit.h"
#include "MagneticField/Engine/interface/MagneticField.h"

class VectorHitMomentumHelper {
public:
  VectorHitMomentumHelper(const MagneticField* magField) {
    GlobalPoint center(0.0, 0.0, 0.0);
    intermediate = magField->inTesla(center).mag() * 0.003;
    //0.003 is because the curvature (rho) is in cm and not in m
  }
  ~VectorHitMomentumHelper() {}

  float transverseMomentum(VectorHit& vh) const {
    float rho = 1. / vh.curvatureORphi(VectorHit::curvatureMode).first;
    return (intermediate * rho);
  }
  float momentum(VectorHit& vh) const { return transverseMomentum(vh) / (1. * sin(vh.theta())); }

private:
  float intermediate;
};

#endif
