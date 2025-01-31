#ifndef RecoTracker_LSTCore_src_alpaka_Triplet_h
#define RecoTracker_LSTCore_src_alpaka_Triplet_h

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"
#include "RecoTracker/LSTCore/interface/TripletsSoA.h"
#include "RecoTracker/LSTCore/interface/Circle.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addTripletToMemory(ModulesConst modules,
                                                         MiniDoubletsConst mds,
                                                         SegmentsConst segments,
                                                         Triplets& triplets,
                                                         unsigned int innerSegmentIndex,
                                                         unsigned int outerSegmentIndex,
                                                         uint16_t innerInnerLowerModuleIndex,
                                                         uint16_t middleLowerModuleIndex,
                                                         uint16_t outerOuterLowerModuleIndex,
#ifdef CUT_VALUE_DEBUG
                                                         float zOut,
                                                         float rtOut,
#endif
                                                         float betaIn,
                                                         float betaInCut,
                                                         float circleRadius,
                                                         float circleCenterX,
                                                         float circleCenterY,
                                                         unsigned int tripletIndex) {
    triplets.segmentIndices()[tripletIndex][0] = innerSegmentIndex;
    triplets.segmentIndices()[tripletIndex][1] = outerSegmentIndex;
    triplets.lowerModuleIndices()[tripletIndex][0] = innerInnerLowerModuleIndex;
    triplets.lowerModuleIndices()[tripletIndex][1] = middleLowerModuleIndex;
    triplets.lowerModuleIndices()[tripletIndex][2] = outerOuterLowerModuleIndex;

    triplets.betaIn()[tripletIndex] = __F2H(betaIn);
    triplets.radius()[tripletIndex] = circleRadius;
    triplets.centerX()[tripletIndex] = circleCenterX;
    triplets.centerY()[tripletIndex] = circleCenterY;
    triplets.logicalLayers()[tripletIndex][0] =
        modules.layers()[innerInnerLowerModuleIndex] + (modules.subdets()[innerInnerLowerModuleIndex] == 4) * 6;
    triplets.logicalLayers()[tripletIndex][1] =
        modules.layers()[middleLowerModuleIndex] + (modules.subdets()[middleLowerModuleIndex] == 4) * 6;
    triplets.logicalLayers()[tripletIndex][2] =
        modules.layers()[outerOuterLowerModuleIndex] + (modules.subdets()[outerOuterLowerModuleIndex] == 4) * 6;
    //get the hits
    unsigned int firstMDIndex = segments.mdIndices()[innerSegmentIndex][0];
    unsigned int secondMDIndex = segments.mdIndices()[innerSegmentIndex][1];
    unsigned int thirdMDIndex = segments.mdIndices()[outerSegmentIndex][1];

    triplets.hitIndices()[tripletIndex][0] = mds.anchorHitIndices()[firstMDIndex];
    triplets.hitIndices()[tripletIndex][1] = mds.outerHitIndices()[firstMDIndex];
    triplets.hitIndices()[tripletIndex][2] = mds.anchorHitIndices()[secondMDIndex];
    triplets.hitIndices()[tripletIndex][3] = mds.outerHitIndices()[secondMDIndex];
    triplets.hitIndices()[tripletIndex][4] = mds.anchorHitIndices()[thirdMDIndex];
    triplets.hitIndices()[tripletIndex][5] = mds.outerHitIndices()[thirdMDIndex];
#ifdef CUT_VALUE_DEBUG
    triplets.zOut()[tripletIndex] = zOut;
    triplets.rtOut()[tripletIndex] = rtOut;
    triplets.betaInCut()[tripletIndex] = betaInCut;
#endif
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passRZConstraint(TAcc const& acc,
                                                       ModulesConst modules,
                                                       MiniDoubletsConst mds,
                                                       uint16_t innerInnerLowerModuleIndex,
                                                       uint16_t middleLowerModuleIndex,
                                                       uint16_t outerOuterLowerModuleIndex,
                                                       unsigned int firstMDIndex,
                                                       unsigned int secondMDIndex,
                                                       unsigned int thirdMDIndex,
                                                       float circleRadius,
                                                       float circleCenterX,
                                                       float circleCenterY) {
    // Using lst_layer numbering convention defined in ModuleMethods.h
    const int layer1 = modules.lstLayers()[innerInnerLowerModuleIndex];
    const int layer2 = modules.lstLayers()[middleLowerModuleIndex];
    const int layer3 = modules.lstLayers()[outerOuterLowerModuleIndex];

    //all the values are stored in the unit of cm, in the calculation below we need to be cautious if we want to use the meter unit
    //get r and z
    const float r1 = mds.anchorRt()[firstMDIndex] / 100;
    const float r2 = mds.anchorRt()[secondMDIndex] / 100;
    const float r3 = mds.anchorRt()[thirdMDIndex] / 100;

    const float z1 = mds.anchorZ()[firstMDIndex] / 100;
    const float z2 = mds.anchorZ()[secondMDIndex] / 100;
    const float z3 = mds.anchorZ()[thirdMDIndex] / 100;

    //use linear approximation for regions 9 and 20-24 because it works better (see https://github.com/SegmentLinking/cmssw/pull/92)
    float residual = alpaka::math::abs(acc, z2 - ((z3 - z1) / (r3 - r1) * (r2 - r1) + z1));

    //region definitions: https://github.com/user-attachments/assets/2b3c1425-66eb-4524-83de-deb6f3b31f71
    if (layer1 == 1 && layer2 == 7) {
      return residual < 0.01f;  // Region 9
    } else if (layer1 == 3 && layer2 == 4) {
      if (layer3 == 5) {
        return residual < 0.037127972f;  // Region 20
      } else if (layer3 == 12) {
        return residual < 0.05f;  // Region 21
      }
    } else if (layer1 == 4) {
      if (layer2 == 12) {
        return residual < 0.063831687f;  // Region 22
      } else if (layer2 == 5) {
        if (layer3 == 6) {
          return residual < 0.04362525f;  // Region 23
        } else if (layer3 == 12) {
          return residual < 0.05f;  // Region 24
        }
      }
    }

    //get the type of module: 0 is ps, 1 is 2s
    const int moduleType3 = modules.moduleType()[outerOuterLowerModuleIndex];

    //get the x,y position of each MD
    const float x1 = mds.anchorX()[firstMDIndex] / 100;
    const float x2 = mds.anchorX()[secondMDIndex] / 100;
    const float x3 = mds.anchorX()[thirdMDIndex] / 100;

    const float y1 = mds.anchorY()[firstMDIndex] / 100;
    const float y2 = mds.anchorY()[secondMDIndex] / 100;
    const float y3 = mds.anchorY()[thirdMDIndex] / 100;

    //set initial and target points
    float x_init = x2;
    float y_init = y2;
    float z_init = z2;
    float r_init = r2;

    float z_target = z3;
    float r_target = r3;

    float x_other = x1;
    float y_other = y1;

    float dz = z2 - z1;

    //use MD2 for regions 5 and 19 because it works better (see https://github.com/SegmentLinking/cmssw/pull/92)
    if ((layer1 == 8 && layer2 == 14 && layer3 == 15) || (layer1 == 3 && layer2 == 12 && layer3 == 13)) {
      x_init = x1;
      y_init = y1;
      z_init = z1;
      r_init = r1;

      z_target = z2;
      r_target = r2;

      x_other = x3;
      y_other = y3;

      dz = z3 - z1;
    }

    //use the 3 MDs to fit a circle. This is the circle parameters, for circle centers and circle radius
    float x_center = circleCenterX / 100;
    float y_center = circleCenterY / 100;
    float pt = 2 * k2Rinv1GeVf * circleRadius;  //k2Rinv1GeVf is already in cm^(-1)

    //determine the charge
    int charge = 0;
    if ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) > 0)
      charge = -1;
    else
      charge = 1;

    //get the absolute value of px and py at the initial point
    float px = 2 * k2Rinv1GeVf * alpaka::math::abs(acc, (y_init - y_center)) * 100;
    float py = 2 * k2Rinv1GeVf * alpaka::math::abs(acc, (x_init - x_center)) * 100;

    //Above line only gives you the correct value of px and py, but signs of px and py calculated below.
    //We look at if the circle is clockwise or anti-clock wise, to make it simpler, we separate the x-y plane into 4 quarters.
    if (x_init > x_center && y_init > y_center)  //1st quad
    {
      if (charge == 1)
        py = -py;
      if (charge == -1)
        px = -px;
    }
    if (x_init < x_center && y_init > y_center)  //2nd quad
    {
      if (charge == -1) {
        px = -px;
        py = -py;
      }
    }
    if (x_init < x_center && y_init < y_center)  //3rd quad
    {
      if (charge == 1)
        px = -px;
      if (charge == -1)
        py = -py;
    }
    if (x_init > x_center && y_init < y_center)  //4th quad
    {
      if (charge == 1) {
        px = -px;
        py = -py;
      }
    }

    //But if the initial T3 curve goes across quarters(i.e. cross axis to separate the quarters), need special redeclaration of px,py signs on these to avoid errors
    if (x3 < x2 && x2 < x1)
      px = -alpaka::math::abs(acc, px);
    else if (x3 > x2 && x2 > x1)
      px = alpaka::math::abs(acc, px);
    if (y3 < y2 && y2 < y1)
      py = -alpaka::math::abs(acc, py);
    else if (y3 > y2 && y2 > y1)
      py = alpaka::math::abs(acc, py);

    float AO = alpaka::math::sqrt(
        acc, (x_other - x_center) * (x_other - x_center) + (y_other - y_center) * (y_other - y_center));
    float BO =
        alpaka::math::sqrt(acc, (x_init - x_center) * (x_init - x_center) + (y_init - y_center) * (y_init - y_center));
    float AB2 = (x_other - x_init) * (x_other - x_init) + (y_other - y_init) * (y_other - y_init);
    float dPhi = alpaka::math::acos(acc, (AO * AO + BO * BO - AB2) / (2 * AO * BO));  //Law of Cosines
    float ds = circleRadius / 100 * dPhi;
    float pz = dz / ds * pt;

    float p = alpaka::math::sqrt(acc, px * px + py * py + pz * pz);
    float a = -2.f * k2Rinv1GeVf * 100 * charge;
    float rou = a / p;

    float rzChiSquared = 0;
    float error = 0;

    //check the tilted module, side: PosZ, NegZ, Center(for not tilted)
    float drdz = alpaka::math::abs(acc, modules.drdzs()[outerOuterLowerModuleIndex]);
    short side = modules.sides()[outerOuterLowerModuleIndex];
    short subdets = modules.subdets()[outerOuterLowerModuleIndex];

    //calculate residual
    if (layer3 <= 6 && ((side == lst::Center) or (drdz < 1))) {  // for barrel
      float paraA = r_init * r_init + 2 * (px * px + py * py) / (a * a) + 2 * (y_init * px - x_init * py) / a -
                    r_target * r_target;
      float paraB = 2 * (x_init * px + y_init * py) / a;
      float paraC = 2 * (y_init * px - x_init * py) / a + 2 * (px * px + py * py) / (a * a);
      float A = paraB * paraB + paraC * paraC;
      float B = 2 * paraA * paraB;
      float C = paraA * paraA - paraC * paraC;
      float sol1 = (-B + alpaka::math::sqrt(acc, B * B - 4 * A * C)) / (2 * A);
      float sol2 = (-B - alpaka::math::sqrt(acc, B * B - 4 * A * C)) / (2 * A);
      float solz1 = alpaka::math::asin(acc, sol1) / rou * pz / p + z_init;
      float solz2 = alpaka::math::asin(acc, sol2) / rou * pz / p + z_init;
      float diffz1 = (solz1 - z_target) * 100;
      float diffz2 = (solz2 - z_target) * 100;
      if (edm::isNotFinite(diffz1))
        residual = diffz2;
      else if (edm::isNotFinite(diffz2))
        residual = diffz1;
      else {
        residual = (alpaka::math::abs(acc, diffz1) < alpaka::math::abs(acc, diffz2)) ? diffz1 : diffz2;
      }
    } else {  // for endcap
      float s = (z_target - z_init) * p / pz;
      float x = x_init + px / a * alpaka::math::sin(acc, rou * s) - py / a * (1 - alpaka::math::cos(acc, rou * s));
      float y = y_init + py / a * alpaka::math::sin(acc, rou * s) + px / a * (1 - alpaka::math::cos(acc, rou * s));
      residual = (r_target - alpaka::math::sqrt(acc, x * x + y * y)) * 100;
    }

    // error
    if (moduleType3 == 0) {
      error = 0.15f;  //PS
    } else {
      error = 5.0f;  //2S
    }

    float projection_missing2 = 1;
    if (drdz < 1)
      projection_missing2 = ((subdets == lst::Endcap) or (side == lst::Center))
                                ? 1.f
                                : 1 / (1 + drdz * drdz);  // cos(atan(drdz)), if dr/dz<1
    if (drdz > 1)
      projection_missing2 = ((subdets == lst::Endcap) or (side == lst::Center))
                                ? 1.f
                                : drdz * drdz / (1 + drdz * drdz);  //sin(atan(drdz)), if dr/dz>1

    rzChiSquared = 12 * (residual * residual) / (error * error * projection_missing2);

    //helix calculation returns NaN, use linear approximation
    if (edm::isNotFinite(rzChiSquared) || circleRadius < 0) {
      float slope = (z3 - z1) / (r3 - r1);

      residual = (layer3 <= 6) ? ((z3 - z1) - slope * (r3 - r1)) : ((r3 - r1) - (z3 - z1) / slope);
      residual = (moduleType3 == 0) ? residual / 0.15f : residual / 5.0f;

      rzChiSquared = 12 * residual * residual;
      return rzChiSquared < 2.8e-4;
    }

    //cuts for different regions
    //region definitions: https://github.com/user-attachments/assets/2b3c1425-66eb-4524-83de-deb6f3b31f71
    //for the logic behind the cuts, see https://github.com/SegmentLinking/cmssw/pull/92
    if (layer1 == 7) {
      if (layer2 == 8) {
        if (layer3 == 9) {
          return rzChiSquared < 65.47191f;  // Region 0
        } else if (layer3 == 14) {
          return rzChiSquared < 3.3200853f;  // Region 1
        }
      } else if (layer2 == 13) {
        return rzChiSquared < 17.194584f;  // Region 2
      }
    } else if (layer1 == 8) {
      if (layer2 == 9) {
        if (layer3 == 10) {
          return rzChiSquared < 114.91959f;  // Region 3
        } else if (layer3 == 15) {
          return rzChiSquared < 3.4359624f;  // Region 4
        }
      } else if (layer2 == 14) {
        return rzChiSquared < 4.6487956f;  // Region 5
      }
    } else if (layer1 == 9) {
      if (layer2 == 10) {
        if (layer3 == 11) {
          return rzChiSquared < 97.34339f;  // Region 6
        } else if (layer3 == 16) {
          return rzChiSquared < 3.095819f;  // Region 7
        }
      } else if (layer2 == 15) {
        return rzChiSquared < 11.477617f;  // Region 8
      }
    } else if (layer1 == 1) {
      if (layer3 == 7) {
        return rzChiSquared < 96.949936f;  // Region 10
      } else if (layer3 == 3) {
        return rzChiSquared < 458.43982f;  // Region 11
      }
    } else if (layer1 == 2) {
      if (layer2 == 7) {
        if (layer3 == 8) {
          return rzChiSquared < 218.82303f;  // Region 12
        } else if (layer3 == 13) {
          return rzChiSquared < 3.155554f;  // Region 13
        }
      } else if (layer2 == 3) {
        if (layer3 == 7) {
          return rzChiSquared < 235.5005f;  // Region 14
        } else if (layer3 == 12) {
          return rzChiSquared < 3.8522234f;  // Region 15
        } else if (layer3 == 4) {
          return rzChiSquared < 3.5852437f;  // Region 16
        }
      }
    } else if (layer1 == 3) {
      if (layer2 == 7) {
        if (layer3 == 8) {
          return rzChiSquared < 42.68f;  // Region 17
        } else if (layer3 == 13) {
          return rzChiSquared < 3.853796f;  // Region 18
        }
      } else if (layer2 == 12) {
        return rzChiSquared < 6.2774787f;  // Region 19
      }
    }
    return false;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPointingConstraintBBB(TAcc const& acc,
                                                                ModulesConst modules,
                                                                MiniDoubletsConst mds,
                                                                SegmentsConst segments,
                                                                uint16_t innerInnerLowerModuleIndex,
                                                                uint16_t middleLowerModuleIndex,
                                                                uint16_t outerOuterLowerModuleIndex,
                                                                unsigned int firstMDIndex,
                                                                unsigned int secondMDIndex,
                                                                unsigned int thirdMDIndex,
                                                                float& zOut,
                                                                float& rtOut,
                                                                unsigned int innerSegmentIndex,
                                                                float& betaIn,
                                                                float& betaInCut,
                                                                const float ptCut) {
    float rtIn = mds.anchorRt()[firstMDIndex];
    float rtMid = mds.anchorRt()[secondMDIndex];
    float drt_InSeg = rtMid - rtIn;

    // raw betaIn value without any correction, based on the mini-doublet hit positions
    float alpha_InLo = __H2F(segments.dPhiChanges()[innerSegmentIndex]);
    float tl_axis_x = mds.anchorX()[thirdMDIndex] - mds.anchorX()[firstMDIndex];
    float tl_axis_y = mds.anchorY()[thirdMDIndex] - mds.anchorY()[firstMDIndex];
    betaIn = alpha_InLo - cms::alpakatools::reducePhiRange(
                              acc, cms::alpakatools::phi(acc, tl_axis_x, tl_axis_y) - mds.anchorPhi()[firstMDIndex]);

    //beta computation
    float drt_tl_axis = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);

    //innerOuterAnchor - innerInnerAnchor
    const float rt_InSeg = alpaka::math::sqrt(acc,
                                              (mds.anchorX()[secondMDIndex] - mds.anchorX()[firstMDIndex]) *
                                                      (mds.anchorX()[secondMDIndex] - mds.anchorX()[firstMDIndex]) +
                                                  (mds.anchorY()[secondMDIndex] - mds.anchorY()[firstMDIndex]) *
                                                      (mds.anchorY()[secondMDIndex] - mds.anchorY()[firstMDIndex]));
    betaInCut =
        alpaka::math::asin(acc, alpaka::math::min(acc, (-rt_InSeg + drt_tl_axis) * k2Rinv1GeVf / ptCut, kSinAlphaMax)) +
        (0.02f / drt_InSeg);

    //Beta cut
    return alpaka::math::abs(acc, betaIn) < betaInCut;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPointingConstraintBBE(TAcc const& acc,
                                                                ModulesConst modules,
                                                                MiniDoubletsConst mds,
                                                                SegmentsConst segments,
                                                                uint16_t innerInnerLowerModuleIndex,
                                                                uint16_t middleLowerModuleIndex,
                                                                uint16_t outerOuterLowerModuleIndex,
                                                                unsigned int firstMDIndex,
                                                                unsigned int secondMDIndex,
                                                                unsigned int thirdMDIndex,
                                                                float& zOut,
                                                                float& rtOut,
                                                                uint16_t innerOuterLowerModuleIndex,
                                                                unsigned int innerSegmentIndex,
                                                                unsigned int outerSegmentIndex,
                                                                float& betaIn,
                                                                float& betaInCut,
                                                                const float ptCut) {
    float rt_InLo = mds.anchorRt()[firstMDIndex];
    float rt_InOut = mds.anchorRt()[secondMDIndex];

    float sdIn_alpha = __H2F(segments.dPhiChanges()[innerSegmentIndex]);

    float tl_axis_x = mds.anchorX()[thirdMDIndex] - mds.anchorX()[firstMDIndex];
    float tl_axis_y = mds.anchorY()[thirdMDIndex] - mds.anchorY()[firstMDIndex];

    betaIn = sdIn_alpha - cms::alpakatools::reducePhiRange(
                              acc, cms::alpakatools::phi(acc, tl_axis_x, tl_axis_y) - mds.anchorPhi()[firstMDIndex]);

    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;

    float swapTemp;

    if (alpaka::math::abs(acc, betaInRHmin) > alpaka::math::abs(acc, betaInRHmax)) {
      swapTemp = betaInRHmin;
      betaInRHmin = betaInRHmax;
      betaInRHmax = swapTemp;
    }

    float sdIn_dr = alpaka::math::sqrt(acc,
                                       (mds.anchorX()[secondMDIndex] - mds.anchorX()[firstMDIndex]) *
                                               (mds.anchorX()[secondMDIndex] - mds.anchorX()[firstMDIndex]) +
                                           (mds.anchorY()[secondMDIndex] - mds.anchorY()[firstMDIndex]) *
                                               (mds.anchorY()[secondMDIndex] - mds.anchorY()[firstMDIndex]));
    float sdIn_d = rt_InOut - rt_InLo;

    float dr = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
    betaInCut = alpaka::math::asin(acc, alpaka::math::min(acc, (-sdIn_dr + dr) * k2Rinv1GeVf / ptCut, kSinAlphaMax)) +
                (0.02f / sdIn_d);

    //Beta cut
    return alpaka::math::abs(acc, betaInRHmin) < betaInCut;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPointingConstraintEEE(TAcc const& acc,
                                                                ModulesConst modules,
                                                                MiniDoubletsConst mds,
                                                                SegmentsConst segments,
                                                                uint16_t innerInnerLowerModuleIndex,
                                                                uint16_t middleLowerModuleIndex,
                                                                uint16_t outerOuterLowerModuleIndex,
                                                                unsigned int firstMDIndex,
                                                                unsigned int secondMDIndex,
                                                                unsigned int thirdMDIndex,
                                                                float& zOut,
                                                                float& rtOut,
                                                                unsigned int innerSegmentIndex,
                                                                unsigned int outerSegmentIndex,
                                                                float& betaIn,
                                                                float& betaInCut,
                                                                const float ptCut) {
    float rt_InLo = mds.anchorRt()[firstMDIndex];
    float rt_InOut = mds.anchorRt()[secondMDIndex];
    float sdIn_alpha = __H2F(segments.dPhiChanges()[innerSegmentIndex]);

    float tl_axis_x = mds.anchorX()[thirdMDIndex] - mds.anchorX()[firstMDIndex];
    float tl_axis_y = mds.anchorY()[thirdMDIndex] - mds.anchorY()[firstMDIndex];

    betaIn = sdIn_alpha - cms::alpakatools::reducePhiRange(
                              acc, cms::alpakatools::phi(acc, tl_axis_x, tl_axis_y) - mds.anchorPhi()[firstMDIndex]);

    float sdIn_alphaRHmin = __H2F(segments.dPhiChangeMins()[innerSegmentIndex]);
    float sdIn_alphaRHmax = __H2F(segments.dPhiChangeMaxs()[innerSegmentIndex]);
    float betaInRHmin = betaIn + sdIn_alphaRHmin - sdIn_alpha;
    float betaInRHmax = betaIn + sdIn_alphaRHmax - sdIn_alpha;

    float swapTemp;

    if (alpaka::math::abs(acc, betaInRHmin) > alpaka::math::abs(acc, betaInRHmax)) {
      swapTemp = betaInRHmin;
      betaInRHmin = betaInRHmax;
      betaInRHmax = swapTemp;
    }
    float sdIn_dr = alpaka::math::sqrt(acc,
                                       (mds.anchorX()[secondMDIndex] - mds.anchorX()[firstMDIndex]) *
                                               (mds.anchorX()[secondMDIndex] - mds.anchorX()[firstMDIndex]) +
                                           (mds.anchorY()[secondMDIndex] - mds.anchorY()[firstMDIndex]) *
                                               (mds.anchorY()[secondMDIndex] - mds.anchorY()[firstMDIndex]));
    float sdIn_d = rt_InOut - rt_InLo;

    float dr = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
    betaInCut = alpaka::math::asin(acc, alpaka::math::min(acc, (-sdIn_dr + dr) * k2Rinv1GeVf / ptCut, kSinAlphaMax)) +
                (0.02f / sdIn_d);

    //Beta cut
    return alpaka::math::abs(acc, betaInRHmin) < betaInCut;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPointingConstraint(TAcc const& acc,
                                                             ModulesConst modules,
                                                             MiniDoubletsConst mds,
                                                             SegmentsConst segments,
                                                             uint16_t innerInnerLowerModuleIndex,
                                                             uint16_t middleLowerModuleIndex,
                                                             uint16_t outerOuterLowerModuleIndex,
                                                             unsigned int firstMDIndex,
                                                             unsigned int secondMDIndex,
                                                             unsigned int thirdMDIndex,
                                                             float& zOut,
                                                             float& rtOut,
                                                             uint16_t innerOuterLowerModuleIndex,
                                                             unsigned int innerSegmentIndex,
                                                             unsigned int outerSegmentIndex,
                                                             float& betaIn,
                                                             float& betaInCut,
                                                             const float ptCut) {
    short innerInnerLowerModuleSubdet = modules.subdets()[innerInnerLowerModuleIndex];
    short middleLowerModuleSubdet = modules.subdets()[middleLowerModuleIndex];
    short outerOuterLowerModuleSubdet = modules.subdets()[outerOuterLowerModuleIndex];

    if (innerInnerLowerModuleSubdet == Barrel and middleLowerModuleSubdet == Barrel and
        outerOuterLowerModuleSubdet == Barrel) {
      return passPointingConstraintBBB(acc,
                                       modules,
                                       mds,
                                       segments,
                                       innerInnerLowerModuleIndex,
                                       middleLowerModuleIndex,
                                       outerOuterLowerModuleIndex,
                                       firstMDIndex,
                                       secondMDIndex,
                                       thirdMDIndex,
                                       zOut,
                                       rtOut,
                                       innerSegmentIndex,
                                       betaIn,
                                       betaInCut,
                                       ptCut);
    } else if (innerInnerLowerModuleSubdet == Barrel and middleLowerModuleSubdet == Barrel and
               outerOuterLowerModuleSubdet == Endcap) {
      return passPointingConstraintBBE(acc,
                                       modules,
                                       mds,
                                       segments,
                                       innerInnerLowerModuleIndex,
                                       middleLowerModuleIndex,
                                       outerOuterLowerModuleIndex,
                                       firstMDIndex,
                                       secondMDIndex,
                                       thirdMDIndex,
                                       zOut,
                                       rtOut,
                                       innerOuterLowerModuleIndex,
                                       innerSegmentIndex,
                                       outerSegmentIndex,
                                       betaIn,
                                       betaInCut,
                                       ptCut);
    } else if (innerInnerLowerModuleSubdet == Barrel and middleLowerModuleSubdet == Endcap and
               outerOuterLowerModuleSubdet == Endcap) {
      return passPointingConstraintBBE(acc,
                                       modules,
                                       mds,
                                       segments,
                                       innerInnerLowerModuleIndex,
                                       middleLowerModuleIndex,
                                       outerOuterLowerModuleIndex,
                                       firstMDIndex,
                                       secondMDIndex,
                                       thirdMDIndex,
                                       zOut,
                                       rtOut,
                                       innerOuterLowerModuleIndex,
                                       innerSegmentIndex,
                                       outerSegmentIndex,
                                       betaIn,
                                       betaInCut,
                                       ptCut);

    }

    else if (innerInnerLowerModuleSubdet == Endcap and middleLowerModuleSubdet == Endcap and
             outerOuterLowerModuleSubdet == Endcap) {
      return passPointingConstraintEEE(acc,
                                       modules,
                                       mds,
                                       segments,
                                       innerInnerLowerModuleIndex,
                                       middleLowerModuleIndex,
                                       outerOuterLowerModuleIndex,
                                       firstMDIndex,
                                       secondMDIndex,
                                       thirdMDIndex,
                                       zOut,
                                       rtOut,
                                       innerSegmentIndex,
                                       outerSegmentIndex,
                                       betaIn,
                                       betaInCut,
                                       ptCut);
    }
    return false;  // failsafe
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runTripletConstraintsAndAlgo(TAcc const& acc,
                                                                   ModulesConst modules,
                                                                   MiniDoubletsConst mds,
                                                                   SegmentsConst segments,
                                                                   uint16_t innerInnerLowerModuleIndex,
                                                                   uint16_t middleLowerModuleIndex,
                                                                   uint16_t outerOuterLowerModuleIndex,
                                                                   unsigned int innerSegmentIndex,
                                                                   unsigned int outerSegmentIndex,
                                                                   float& zOut,
                                                                   float& rtOut,
                                                                   float& betaIn,
                                                                   float& betaInCut,
                                                                   float& circleRadius,
                                                                   float& circleCenterX,
                                                                   float& circleCenterY,
                                                                   const float ptCut) {
    //this cut reduces the number of candidates by a factor of 4, i.e., 3 out of 4 warps can end right here!
    if (segments.mdIndices()[innerSegmentIndex][1] != segments.mdIndices()[outerSegmentIndex][0])
      return false;

    unsigned int firstMDIndex = segments.mdIndices()[innerSegmentIndex][0];
    unsigned int secondMDIndex = segments.mdIndices()[outerSegmentIndex][0];
    unsigned int thirdMDIndex = segments.mdIndices()[outerSegmentIndex][1];

    float x1 = mds.anchorX()[firstMDIndex];
    float x2 = mds.anchorX()[secondMDIndex];
    float x3 = mds.anchorX()[thirdMDIndex];
    float y1 = mds.anchorY()[firstMDIndex];
    float y2 = mds.anchorY()[secondMDIndex];
    float y3 = mds.anchorY()[thirdMDIndex];

    std::tie(circleRadius, circleCenterX, circleCenterY) =
        computeRadiusFromThreeAnchorHits(acc, x1, y1, x2, y2, x3, y3);

    if (not passRZConstraint(acc,
                             modules,
                             mds,
                             innerInnerLowerModuleIndex,
                             middleLowerModuleIndex,
                             outerOuterLowerModuleIndex,
                             firstMDIndex,
                             secondMDIndex,
                             thirdMDIndex,
                             circleRadius,
                             circleCenterX,
                             circleCenterY))
      return false;

    if (not passPointingConstraint(acc,
                                   modules,
                                   mds,
                                   segments,
                                   innerInnerLowerModuleIndex,
                                   middleLowerModuleIndex,
                                   outerOuterLowerModuleIndex,
                                   firstMDIndex,
                                   secondMDIndex,
                                   thirdMDIndex,
                                   zOut,
                                   rtOut,
                                   middleLowerModuleIndex,
                                   innerSegmentIndex,
                                   outerSegmentIndex,
                                   betaIn,
                                   betaInCut,
                                   ptCut))
      return false;

    return true;
  }

  struct CreateTriplets {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  ModulesConst modules,
                                  MiniDoubletsConst mds,
                                  SegmentsConst segments,
                                  SegmentsOccupancyConst segmentsOccupancy,
                                  Triplets triplets,
                                  TripletsOccupancy tripletsOccupancy,
                                  ObjectRangesConst ranges,
                                  uint16_t* index_gpu,
                                  uint16_t nonZeroModules,
                                  const float ptCut) const {
      for (uint16_t innerLowerModuleArrayIdx : cms::alpakatools::uniform_elements_z(acc, nonZeroModules)) {
        uint16_t innerInnerLowerModuleIndex = index_gpu[innerLowerModuleArrayIdx];
        if (innerInnerLowerModuleIndex >= modules.nLowerModules())
          continue;

        uint16_t nConnectedModules = modules.nConnectedModules()[innerInnerLowerModuleIndex];
        if (nConnectedModules == 0)
          continue;

        unsigned int nInnerSegments = segmentsOccupancy.nSegments()[innerInnerLowerModuleIndex];
        for (unsigned int innerSegmentArrayIndex : cms::alpakatools::uniform_elements_y(acc, nInnerSegments)) {
          unsigned int innerSegmentIndex =
              ranges.segmentRanges()[innerInnerLowerModuleIndex][0] + innerSegmentArrayIndex;

          // middle lower module - outer lower module of inner segment
          uint16_t middleLowerModuleIndex = segments.outerLowerModuleIndices()[innerSegmentIndex];

          unsigned int nOuterSegments = segmentsOccupancy.nSegments()[middleLowerModuleIndex];
          for (unsigned int outerSegmentArrayIndex : cms::alpakatools::uniform_elements_x(acc, nOuterSegments)) {
            unsigned int outerSegmentIndex = ranges.segmentRanges()[middleLowerModuleIndex][0] + outerSegmentArrayIndex;

            uint16_t outerOuterLowerModuleIndex = segments.outerLowerModuleIndices()[outerSegmentIndex];

            float zOut, rtOut, betaIn, betaInCut, circleRadius, circleCenterX, circleCenterY;

            bool success = runTripletConstraintsAndAlgo(acc,
                                                        modules,
                                                        mds,
                                                        segments,
                                                        innerInnerLowerModuleIndex,
                                                        middleLowerModuleIndex,
                                                        outerOuterLowerModuleIndex,
                                                        innerSegmentIndex,
                                                        outerSegmentIndex,
                                                        zOut,
                                                        rtOut,
                                                        betaIn,
                                                        betaInCut,
                                                        circleRadius,
                                                        circleCenterX,
                                                        circleCenterY,
                                                        ptCut);

            if (success) {
              unsigned int totOccupancyTriplets =
                  alpaka::atomicAdd(acc,
                                    &tripletsOccupancy.totOccupancyTriplets()[innerInnerLowerModuleIndex],
                                    1u,
                                    alpaka::hierarchy::Threads{});
              if (static_cast<int>(totOccupancyTriplets) >=
                  ranges.tripletModuleOccupancy()[innerInnerLowerModuleIndex]) {
#ifdef WARNINGS
                printf("Triplet excess alert! Module index = %d, Occupancy = %d\n",
                       innerInnerLowerModuleIndex,
                       totOccupancyTriplets);
#endif
              } else {
                unsigned int tripletModuleIndex = alpaka::atomicAdd(
                    acc, &tripletsOccupancy.nTriplets()[innerInnerLowerModuleIndex], 1u, alpaka::hierarchy::Threads{});
                unsigned int tripletIndex =
                    ranges.tripletModuleIndices()[innerInnerLowerModuleIndex] + tripletModuleIndex;
                addTripletToMemory(modules,
                                   mds,
                                   segments,
                                   triplets,
                                   innerSegmentIndex,
                                   outerSegmentIndex,
                                   innerInnerLowerModuleIndex,
                                   middleLowerModuleIndex,
                                   outerOuterLowerModuleIndex,
#ifdef CUT_VALUE_DEBUG
                                   zOut,
                                   rtOut,
#endif
                                   betaIn,
                                   betaInCut,
                                   circleRadius,
                                   circleCenterX,
                                   circleCenterY,
                                   tripletIndex);
              }
            }
          }
        }
      }
    }
  };

  struct CreateTripletArrayRanges {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  ObjectRanges ranges,
                                  SegmentsOccupancyConst segmentsOccupancy,
                                  const float ptCut) const {
      // implementation is 1D with a single block
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      // Initialize variables in shared memory and set to 0
      int& nTotalTriplets = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        nTotalTriplets = 0;
      }
      alpaka::syncBlockThreads(acc);

      // Occupancy matrix for 0.8 GeV pT Cut
      constexpr int p08_occupancy_matrix[4][4] = {
          {543, 235, 88, 46},  // category 0
          {755, 347, 0, 0},    // category 1
          {0, 0, 0, 0},        // category 2
          {0, 38, 46, 39}      // category 3
      };

      // Occupancy matrix for 0.6 GeV pT Cut, 99.9%
      constexpr int p06_occupancy_matrix[4][4] = {
          {1146, 544, 216, 83},  // category 0
          {1032, 275, 0, 0},     // category 1
          {0, 0, 0, 0},          // category 2
          {0, 115, 110, 76}      // category 3
      };

      // Select the appropriate occupancy matrix based on ptCut
      const auto& occupancy_matrix = (ptCut < 0.8f) ? p06_occupancy_matrix : p08_occupancy_matrix;

      for (uint16_t i : cms::alpakatools::uniform_elements(acc, modules.nLowerModules())) {
        if (segmentsOccupancy.nSegments()[i] == 0) {
          ranges.tripletModuleIndices()[i] = nTotalTriplets;
          ranges.tripletModuleOccupancy()[i] = 0;
          continue;
        }

        short module_rings = modules.rings()[i];
        short module_layers = modules.layers()[i];
        short module_subdets = modules.subdets()[i];
        float module_eta = alpaka::math::abs(acc, modules.eta()[i]);

        int category_number = getCategoryNumber(module_layers, module_subdets, module_rings);
        int eta_number = getEtaBin(module_eta);

        int occupancy = 0;
        if (category_number != -1 && eta_number != -1) {
          occupancy = occupancy_matrix[category_number][eta_number];
        }
#ifdef WARNINGS
        else {
          printf("Unhandled case in createTripletArrayRanges! Module index = %i\n", i);
        }
#endif

        ranges.tripletModuleOccupancy()[i] = occupancy;
        unsigned int nTotT = alpaka::atomicAdd(acc, &nTotalTriplets, occupancy, alpaka::hierarchy::Threads{});
        ranges.tripletModuleIndices()[i] = nTotT;
      }

      // Wait for all threads to finish before reporting final values
      alpaka::syncBlockThreads(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        ranges.nTotalTrips() = nTotalTriplets;
      }
    }
  };

  struct AddTripletRangesToEventExplicit {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  TripletsOccupancyConst tripletsOccupancy,
                                  ObjectRanges ranges) const {
      // implementation is 1D with a single block
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      for (uint16_t i : cms::alpakatools::uniform_elements(acc, modules.nLowerModules())) {
        if (tripletsOccupancy.nTriplets()[i] == 0) {
          ranges.tripletRanges()[i][0] = -1;
          ranges.tripletRanges()[i][1] = -1;
        } else {
          ranges.tripletRanges()[i][0] = ranges.tripletModuleIndices()[i];
          ranges.tripletRanges()[i][1] = ranges.tripletModuleIndices()[i] + tripletsOccupancy.nTriplets()[i] - 1;
        }
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
