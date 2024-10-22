#include "CondFormats/ESObjects/interface/ESRecHitRatioCuts.h"

ESRecHitRatioCuts::ESRecHitRatioCuts() {
  r12Low_ = 0.;
  r12High_ = 0.;
  r23Low_ = 0.;
  r23High_ = 0.;
}

ESRecHitRatioCuts::ESRecHitRatioCuts(const float& r12Low,
                                     const float& r23Low,
                                     const float& r12High,
                                     const float& r23High) {
  r12Low_ = r12Low;
  r12High_ = r12High;
  r23Low_ = r23Low;
  r23High_ = r23High;
}

ESRecHitRatioCuts::~ESRecHitRatioCuts() {}
