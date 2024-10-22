#include "CalibTracker/SiStripHitResolution/interface/SiStripOverlapHit.h"
#include <cmath>

SiStripOverlapHit::SiStripOverlapHit(TrajectoryMeasurement const& measA, TrajectoryMeasurement const& measB) {
  // check which hit is closer to the IP
  // assign it to hitA_, the other to hitB_
  double rA = measA.recHit()->globalPosition().perp();
  double rB = measB.recHit()->globalPosition().perp();
  if (rA < rB) {
    measA_ = measA;
    measB_ = measB;
  } else {
    measA_ = measB;
    measB_ = measA;
  }
}

TrajectoryStateOnSurface const& SiStripOverlapHit::trajectoryStateOnSurface(unsigned int hit, bool updated) const {
  assert(hit < 2);
  switch (hit) {
    case 0:
      return updated ? measA_.updatedState() : measA_.predictedState();
    case 1:
      return updated ? measB_.updatedState() : measB_.predictedState();
    default:
      return measA_.updatedState();
  }
}

double SiStripOverlapHit::getTrackLocalAngle(unsigned int hit) const {
  // since x is the precise coordinate and z is pointing out, we want the angle between z and x
  return hit ? atan(trajectoryStateOnSurface(hit - 1).localDirection().x() /
                    trajectoryStateOnSurface(hit - 1).localDirection().z())
             : (getTrackLocalAngle(1) + getTrackLocalAngle(2)) / 2.;
}

double SiStripOverlapHit::offset(unsigned int hit) const {
  assert(hit < 2);
  // x is the precise coordinate
  return this->hit(hit)->localPosition().x() - trajectoryStateOnSurface(hit, false).localPosition().x();
}

double SiStripOverlapHit::shift() const {
  // so this is the double difference
  return offset(0) - offset(1);
}

double SiStripOverlapHit::distance(bool fromTrajectory) const {
  if (fromTrajectory) {
    return (trajectoryStateOnSurface(0, true).globalPosition().basicVector() -
            trajectoryStateOnSurface(1, true).globalPosition().basicVector())
        .mag();
  } else {
    return (hitA()->globalPosition().basicVector() - hitB()->globalPosition().basicVector()).mag();
  }
}

GlobalPoint SiStripOverlapHit::position() const {
  auto vector = (hitA()->globalPosition().basicVector() + hitB()->globalPosition().basicVector()) / 2.;
  return GlobalPoint(vector);
}
