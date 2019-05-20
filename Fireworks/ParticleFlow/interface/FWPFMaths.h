#ifndef _FWPFMATHS_H_
#define _FWPFMATHS_H_

#include "TEveVector.h"

namespace FWPFMaths {
  TEveVector lineCircleIntersect(const TEveVector &v1, const TEveVector &v2, float r);
  TEveVector lineLineIntersect(const TEveVector &v1, const TEveVector &v2, const TEveVector &v3, const TEveVector &v4);
  TEveVector cross(const TEveVector &v1, const TEveVector &v2);
  TEveVector calculateCentre(const float *vertices);
  float linearInterpolation(const TEveVector &p1, const TEveVector &p2, float r);
  float dot(const TEveVector &v1, const TEveVector &v2);
  float sgn(float val);
  float calculateEt(const TEveVector &centre, float e);
  bool checkIntersect(const TEveVector &vec, float r);
}  // namespace FWPFMaths
#endif
