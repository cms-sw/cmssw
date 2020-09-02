#ifndef VectorHitMomentumHelper_H
#define VectorHitMomentumHelper_H

#include "DataFormats/TrackerRecHit2D/interface/VectorHit.h"
#include "MagneticField/Engine/interface/MagneticField.h"

class VectorHitMomentumHelper {
public:
  VectorHitMomentumHelper();
  ~VectorHitMomentumHelper() {
  }

  float transverseMomentum(VectorHit& vh,const MagneticField* magField){
	  GlobalPoint center(0.0, 0.0, 0.0);
	  float magnT = magField->inTesla(center).mag();
	  double rho = 1. / vh.curvatureORphi("curvature").first;
	  //0.003 is because the curvature (rho) is in cm and not in m
	   return (0.003 * magnT * rho);
  }
  float momentum(VectorHit& vh, const MagneticField* magField) { return transverseMomentum(vh , magField) / (1. * sin(vh.theta())); }

private:
};

#endif
