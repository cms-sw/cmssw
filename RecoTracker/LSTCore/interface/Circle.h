#ifndef RecoTracker_LSTCore_interface_Circle_h
#define RecoTracker_LSTCore_interface_Circle_h

#include <tuple>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace lst {

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE std::tuple<float, float, float> computeRadiusFromThreeAnchorHits(
      TAcc const& acc, float x1, float y1, float x2, float y2, float x3, float y3) {
    float radius = 0.f;

    //first anchor hit - (x1,y1), second anchor hit - (x2,y2), third anchor hit - (x3, y3)

    float denomInv = 1.0f / ((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3));

    float xy1sqr = x1 * x1 + y1 * y1;

    float xy2sqr = x2 * x2 + y2 * y2;

    float xy3sqr = x3 * x3 + y3 * y3;

    float regressionCenterX = 0.5f * ((y3 - y2) * xy1sqr + (y1 - y3) * xy2sqr + (y2 - y1) * xy3sqr) * denomInv;

    float regressionCenterY = 0.5f * ((x2 - x3) * xy1sqr + (x3 - x1) * xy2sqr + (x1 - x2) * xy3sqr) * denomInv;

    float c = ((x2 * y3 - x3 * y2) * xy1sqr + (x3 * y1 - x1 * y3) * xy2sqr + (x1 * y2 - x2 * y1) * xy3sqr) * denomInv;

    if (((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3) == 0) ||
        (regressionCenterX * regressionCenterX + regressionCenterY * regressionCenterY - c < 0)) {
#ifdef WARNINGS
      printf("three collinear points or FATAL! r^2 < 0!\n");
#endif
      radius = -1.f;
    } else
      radius =
          alpaka::math::sqrt(acc, regressionCenterX * regressionCenterX + regressionCenterY * regressionCenterY - c);

    return std::make_tuple(radius, regressionCenterX, regressionCenterY);
  }

}  //namespace lst

#endif
