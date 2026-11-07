#ifndef RecoTracker_LSTCore_src_alpaka_Triplet_h
#define RecoTracker_LSTCore_src_alpaka_Triplet_h

#include <limits>

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsSoA.h"
#include "RecoTracker/LSTCore/interface/SegmentsSoA.h"
#include "RecoTracker/LSTCore/interface/TripletsSoA.h"
#include "RecoTracker/LSTCore/interface/Circle.h"

#include "NeuralNetwork.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  // Pre-loaded inner-segment-constant data for passPointingConstraint.
  // Populated once per inner segment, reused across all outer segments in the inner loop.
  struct T3InnerSegData {
    float x1, y1;           // first MD position (cm)
    float x2, y2;           // second MD position (shared MD, cm)
    float rt1, rt2;         // anchorRt for first two MDs (cm)
    float drt_InSeg;        // rt2 - rt1
    float rt_InSeg;         // sqrt((x2-x1)^2 + (y2-y1)^2)
    float sdIn_alpha;       // dPhiChange of inner segment
    float sdIn_alphaRHmin;  // dPhiChangeMin (for endcap path)
    float sdIn_alphaRHmax;  // dPhiChangeMax (for endcap path)
    // Precomputed sin/cos for algebraic betaIn check (avoids per-candidate atan2).
    float sin_alpha, cos_alpha;
    float sin_alphaRHmin, cos_alphaRHmin;  // for endcap EEE path
    float sin_alphaRHmax, cos_alphaRHmax;  // for endcap EEE path
    short innerSubdet;                     // subdet of inner-inner module
    short middleSubdet;                    // subdet of middle module
  };

  // Pre-loaded hit coordinates for passRZConstraint.
  // All values in cm, passRZConstraint converts to meters internally.
  struct T3HitCoords {
    float x1, y1, z1, rt1;
    float x2, y2, z2, rt2;
    float x3, y3, z3, rt3;
  };

  // Triplet charge: the bending direction of the three anchor hits in the transverse plane.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE short tripletCharge(T3HitCoords const& hitCoords) {
    const float cross = (hitCoords.x2 - hitCoords.x1) * (hitCoords.y3 - hitCoords.y1) -
                        (hitCoords.y2 - hitCoords.y1) * (hitCoords.x3 - hitCoords.x1);
    return -1 * ((int)copysignf(1.0f, cross));
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE T3InnerSegData loadT3InnerSegData(TAcc const& acc,
                                                                   MiniDoubletsConst mds,
                                                                   SegmentsConst segments,
                                                                   ModulesConst modules,
                                                                   unsigned int innerSegmentIndex,
                                                                   uint16_t innerInnerLowerModuleIndex,
                                                                   uint16_t middleLowerModuleIndex) {
    unsigned int firstMDIndex = segments.mdIndices()[innerSegmentIndex][0];
    unsigned int secondMDIndex = segments.mdIndices()[innerSegmentIndex][1];
    T3InnerSegData d;
    d.x1 = mds.anchorX()[firstMDIndex];
    d.y1 = mds.anchorY()[firstMDIndex];
    d.x2 = mds.anchorX()[secondMDIndex];
    d.y2 = mds.anchorY()[secondMDIndex];
    d.rt1 = mds.anchorRt()[firstMDIndex];
    d.rt2 = mds.anchorRt()[secondMDIndex];
    d.drt_InSeg = d.rt2 - d.rt1;
    d.rt_InSeg = alpaka::math::sqrt(acc, (d.x2 - d.x1) * (d.x2 - d.x1) + (d.y2 - d.y1) * (d.y2 - d.y1));
    d.sdIn_alpha = __H2F(segments.dPhiChanges()[innerSegmentIndex]);
    d.sin_alpha = alpaka::math::sin(acc, d.sdIn_alpha);
    d.cos_alpha = alpaka::math::cos(acc, d.sdIn_alpha);
    d.innerSubdet = modules.subdets()[innerInnerLowerModuleIndex];
    d.middleSubdet = modules.subdets()[middleLowerModuleIndex];
    d.sdIn_alphaRHmin = __H2F(segments.dPhiChangeMins()[innerSegmentIndex]);
    d.sdIn_alphaRHmax = __H2F(segments.dPhiChangeMaxs()[innerSegmentIndex]);
    if (d.innerSubdet == Endcap and d.middleSubdet == Endcap) {
      d.sin_alphaRHmin = alpaka::math::sin(acc, d.sdIn_alphaRHmin);
      d.cos_alphaRHmin = alpaka::math::cos(acc, d.sdIn_alphaRHmin);
      d.sin_alphaRHmax = alpaka::math::sin(acc, d.sdIn_alphaRHmax);
      d.cos_alphaRHmax = alpaka::math::cos(acc, d.sdIn_alphaRHmax);
    }
    return d;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addTripletToMemory(TripletsBuild triplets,
                                                         unsigned int innerSegmentIndex,
                                                         unsigned int outerSegmentIndex,
                                                         float betaIn,
                                                         float betaInCut,
                                                         float circleRadius,
                                                         float circleCenterX,
                                                         float circleCenterY,
                                                         unsigned int tripletIndex,
                                                         float (&t3Scores)[dnn::t3dnn::kOutputFeatures],
                                                         short charge,
                                                         uint8_t flags) {
    triplets.segmentIndices()[tripletIndex][0] = innerSegmentIndex;
    triplets.segmentIndices()[tripletIndex][1] = outerSegmentIndex;

    triplets.radius()[tripletIndex] = circleRadius;
    triplets.centerX()[tripletIndex] = circleCenterX;
    triplets.centerY()[tripletIndex] = circleCenterY;
    triplets.charge()[tripletIndex] = static_cast<int8_t>(charge);
    triplets.flags()[tripletIndex] = flags;
#ifdef CUT_VALUE_DEBUG
    triplets.betaIn()[tripletIndex] = __F2H(betaIn);
    triplets.betaInCut()[tripletIndex] = betaInCut;
#endif

    triplets.fakeScore()[tripletIndex] = t3Scores[0];
    triplets.promptScore()[tripletIndex] = t3Scores[1];
    triplets.displacedScore()[tripletIndex] = t3Scores[2];
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passRZConstraint(TAcc const& acc,
                                                       ModulesConst modules,
                                                       uint16_t innerInnerLowerModuleIndex,
                                                       uint16_t middleLowerModuleIndex,
                                                       uint16_t outerOuterLowerModuleIndex,
                                                       T3HitCoords const& hitCoords,
                                                       float circleRadius,
                                                       float circleCenterX,
                                                       float circleCenterY,
                                                       short& charge,
                                                       const float cutScale = 1.f) {
    // Using lst_layer numbering convention defined in ModuleMethods.h
    const short layer1 = modules.lstLayers()[innerInnerLowerModuleIndex];
    const short layer2 = modules.lstLayers()[middleLowerModuleIndex];
    const short layer3 = modules.lstLayers()[outerOuterLowerModuleIndex];

    //all the values are stored in the unit of cm, in the calculation below we need to be cautious if we want to use the meter unit
    //get r and z (convert from cm to m)
    const float r1 = hitCoords.rt1 / 100;
    const float r2 = hitCoords.rt2 / 100;
    const float r3 = hitCoords.rt3 / 100;

    const float z1 = hitCoords.z1 / 100;
    const float z2 = hitCoords.z2 / 100;
    const float z3 = hitCoords.z3 / 100;

    //use linear approximation for regions 9 and 20-24 because it works better (see https://github.com/SegmentLinking/cmssw/pull/92)
    float residual = alpaka::math::abs(acc, z2 - ((z3 - z1) / (r3 - r1) * (r2 - r1) + z1));

    //get the x,y position of each MD (convert from cm to m)
    const float x1 = hitCoords.x1 / 100;
    const float x2 = hitCoords.x2 / 100;
    const float x3 = hitCoords.x3 / 100;

    const float y1 = hitCoords.y1 / 100;
    const float y2 = hitCoords.y2 / 100;
    const float y3 = hitCoords.y3 / 100;

    charge = tripletCharge(hitCoords);

    //region definitions: https://github.com/user-attachments/assets/2b3c1425-66eb-4524-83de-deb6f3b31f71
    if (layer1 == 1 && layer2 == 7) {
      return residual < 0.01f * cutScale;  // Region 9
    } else if (layer1 == 3 && layer2 == 4) {
      if (layer3 == 5) {
        return residual < 0.037127972f * cutScale;  // Region 20
      } else if (layer3 == 12) {
        return residual < 0.05f * cutScale;  // Region 21
      }
    } else if (layer1 == 4) {
      if (layer2 == 12) {
        return residual < 0.063831687f * cutScale;  // Region 22
      } else if (layer2 == 5) {
        if (layer3 == 6) {
          return residual < 0.04362525f * cutScale;  // Region 23
        } else if (layer3 == 12) {
          return residual < 0.05f * cutScale;  // Region 24
        }
      }
    }

    //get the type of module: 0 is ps, 1 is 2s
    const bool moduleType3 = modules.moduleType()[outerOuterLowerModuleIndex];

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

    //get the px and py at the initial point
    float px = 2 * charge * k2Rinv1GeVf * (y_init - y_center) * 100;
    float py = -2 * charge * k2Rinv1GeVf * (x_init - x_center) * 100;

    //But if the initial T3 curve goes across quarters(i.e. cross axis to separate the quarters), need special redeclaration of px,py signs on these to avoid errors
    if (x3 < x2 && x2 < x1)
      px = -alpaka::math::abs(acc, px);
    else if (x3 > x2 && x2 > x1)
      px = alpaka::math::abs(acc, px);
    if (y3 < y2 && y2 < y1)
      py = -alpaka::math::abs(acc, py);
    else if (y3 > y2 && y2 > y1)
      py = alpaka::math::abs(acc, py);

    // All 3 hits lie on the fitted circle, so AO = BO = R; eliminates 2 sqrt calls.
    float R = circleRadius / 100;
    float AB2 = (x_other - x_init) * (x_other - x_init) + (y_other - y_init) * (y_other - y_init);
    float dPhi = alpaka::math::acos(acc, 1 - AB2 / (2 * R * R));
    float ds = R * dPhi;
    float pz = dz / ds * pt;

    float p = alpaka::math::sqrt(acc, px * px + py * py + pz * pz);
    float a = -2.f * k2Rinv1GeVf * 100 * charge;
    float rou = a / p;

    float rzChiSquared = 0;
    float error = 0;

    //check the tilted module, side: PosZ, NegZ, Center(for not tilted)
    float drdz = alpaka::math::abs(acc, modules.drdzs()[outerOuterLowerModuleIndex]);
    const short side = modules.sides()[outerOuterLowerModuleIndex];
    const short subdets = modules.subdets()[outerOuterLowerModuleIndex];

    //calculate residual
    if (layer3 <= 6 && ((side == lst::Center) or (drdz < 1))) {  // for barrel
      float paraA = r_init * r_init + 2 * (px * px + py * py) / (a * a) + 2 * (y_init * px - x_init * py) / a -
                    r_target * r_target;
      float paraB = 2 * (x_init * px + y_init * py) / a;
      float paraC = 2 * (y_init * px - x_init * py) / a + 2 * (px * px + py * py) / (a * a);
      float A = paraB * paraB + paraC * paraC;
      float B = 2 * paraA * paraB;
      float C = paraA * paraA - paraC * paraC;
      // Shared discriminant: compute sqrt once instead of twice.
      float disc = alpaka::math::sqrt(acc, B * B - 4 * A * C);
      float sol1 = (-B + disc) / (2 * A);
      float sol2 = (-B - disc) / (2 * A);
      float solz1 = alpaka::math::asin(acc, sol1) / rou * pz / p + z_init;
      float solz2 = alpaka::math::asin(acc, sol2) / rou * pz / p + z_init;
      float diffz1 = (solz1 - z_target) * 100;
      float diffz2 = (solz2 - z_target) * 100;
      residual = edm::isNotFinite(diffz1) ? diffz2
                 : edm::isNotFinite(diffz2)
                     ? diffz1
                     : ((alpaka::math::abs(acc, diffz1) < alpaka::math::abs(acc, diffz2)) ? diffz1 : diffz2);
    } else {  // for endcap
      float s = (z_target - z_init) * p / pz;
      // Shared sin/cos: compute once instead of twice each.
      float sinRS = alpaka::math::sin(acc, rou * s);
      float cosRS = alpaka::math::cos(acc, rou * s);
      float x = x_init + px / a * sinRS - py / a * (1 - cosRS);
      float y = y_init + py / a * sinRS + px / a * (1 - cosRS);
      residual = (r_target - alpaka::math::sqrt(acc, x * x + y * y)) * 100;
    }

    // error, PS layer uncertainty is 0.15cm, 2S uncertainty is 5cm.
    error = moduleType3 == 0 ? 0.15f : 5.0f;

    const bool isEndcapOrCenter = (subdets == lst::Endcap) or (side == lst::Center);
    float projection_missing2 = 1;
    if (drdz < 1)
      projection_missing2 = isEndcapOrCenter ? 1.f : 1 / (1 + drdz * drdz);  // cos(atan(drdz)), if dr/dz<1
    if (drdz > 1)
      projection_missing2 = isEndcapOrCenter ? 1.f : drdz * drdz / (1 + drdz * drdz);  //sin(atan(drdz)), if dr/dz>1

    rzChiSquared = 12 * (residual * residual) / (error * error * projection_missing2);

    // A loosened copy of this cut keeps every candidate whose helix residual is not finite.
    if (cutScale > 1.f && edm::isNotFinite(rzChiSquared))
      return true;
    //helix calculation returns NaN, use linear approximation
    if (edm::isNotFinite(rzChiSquared) || circleRadius < 0) {
      float slope = (z3 - z1) / (r3 - r1);

      residual = (layer3 <= 6) ? ((z3 - z1) - slope * (r3 - r1)) : ((r3 - r1) - (z3 - z1) / slope);
      residual = (moduleType3 == 0) ? residual / 0.15f : residual / 5.0f;

      rzChiSquared = 12 * residual * residual;
      return rzChiSquared < 2.8e-4 * cutScale;
    }

    //cuts for different regions
    //region definitions: https://github.com/user-attachments/assets/2b3c1425-66eb-4524-83de-deb6f3b31f71
    //for the logic behind the cuts, see https://github.com/SegmentLinking/cmssw/pull/92
    if (layer1 == 7) {
      if (layer2 == 8) {
        if (layer3 == 9) {
          return rzChiSquared < 65.47191f * cutScale;  // Region 0
        } else if (layer3 == 14) {
          return rzChiSquared < 3.3200853f * cutScale;  // Region 1
        }
      } else if (layer2 == 13) {
        return rzChiSquared < 17.194584f * cutScale;  // Region 2
      }
    } else if (layer1 == 8) {
      if (layer2 == 9) {
        if (layer3 == 10) {
          return rzChiSquared < 114.91959f * cutScale;  // Region 3
        } else if (layer3 == 15) {
          return rzChiSquared < 3.4359624f * cutScale;  // Region 4
        }
      } else if (layer2 == 14) {
        return rzChiSquared < 4.6487956f * cutScale;  // Region 5
      }
    } else if (layer1 == 9) {
      if (layer2 == 10) {
        if (layer3 == 11) {
          return rzChiSquared < 97.34339f * cutScale;  // Region 6
        } else if (layer3 == 16) {
          return rzChiSquared < 3.095819f * cutScale;  // Region 7
        }
      } else if (layer2 == 15) {
        return rzChiSquared < 11.477617f * cutScale;  // Region 8
      }
    } else if (layer1 == 1) {
      if (layer3 == 7) {
        return rzChiSquared < 96.949936f * cutScale;  // Region 10
      } else if (layer3 == 3) {
        return rzChiSquared < 458.43982f * cutScale;  // Region 11
      }
    } else if (layer1 == 2) {
      if (layer2 == 7) {
        if (layer3 == 8) {
          return rzChiSquared < 218.82303f * cutScale;  // Region 12
        } else if (layer3 == 13) {
          return rzChiSquared < 3.155554f * cutScale;  // Region 13
        }
      } else if (layer2 == 3) {
        if (layer3 == 7) {
          return rzChiSquared < 235.5005f * cutScale;  // Region 14
        } else if (layer3 == 12) {
          return rzChiSquared < 3.8522234f * cutScale;  // Region 15
        } else if (layer3 == 4) {
          return rzChiSquared < 3.5852437f * cutScale;  // Region 16
        }
      }
    } else if (layer1 == 3) {
      if (layer2 == 7) {
        if (layer3 == 8) {
          return rzChiSquared < 42.68f * cutScale;  // Region 17
        } else if (layer3 == 13) {
          return rzChiSquared < 3.853796f * cutScale;  // Region 18
        }
      } else if (layer2 == 12) {
        return rzChiSquared < 6.2774787f * cutScale;  // Region 19
      }
    }
    return false;
  }

  // Returns 0 if the pointing constraint fails, 1 if it passes, 2 if it passes only the widened bound.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int passPointingConstraint(
      TAcc const& acc, T3InnerSegData const& innerSegData, float x3, float y3, short outerSubdet, const float ptCut) {
    const float dx = x3 - innerSegData.x1;
    const float dy = y3 - innerSegData.y1;
    const float drt_tl_axis = alpaka::math::sqrt(acc, dx * dx + dy * dy);

    const float betaInCut =
        alpaka::math::asin(
            acc, alpaka::math::min(acc, (-innerSegData.rt_InSeg + drt_tl_axis) * k2Rinv1GeVf / ptCut, kSinAlphaMax)) +
        (0.02f / innerSegData.drt_InSeg);

    // Algebraic betaIn check, avoiding per-candidate atan2.
    // betaIn = sdIn_alpha - (phi(dx,dy) - anchorPhi1)
    //        = sdIn_alpha - atan2(x1*y3 - y1*x3, x1*x3 + y1*y3 - rt1^2)
    // Let a = sdIn_alpha, b = atan2(cross, dot) where cross and dot are the
    // 2D cross/dot products of r1 with the (hit1->hit3) displacement, so betaIn = a - b.
    // Using sin(a-b) = sin(a)*cos(b) - cos(a)*sin(b) with sin(b)=cross/r, cos(b)=dot/r:
    //   sinBetaIn = (sin(a)*dot - cos(a)*cross) / r,  r = sqrt(cross^2 + dot^2)
    // sin(a)/cos(a) are precomputed in T3InnerSegData to avoid per-candidate trig.
    // The cut |betaIn| < betaInCut becomes sinBetaIn^2 < sin(betaInCut)^2 * r^2,
    // with a cos(betaIn) > 0 sign check (from cos(a-b) identity)
    const float crossBetaIn = innerSegData.x1 * y3 - innerSegData.y1 * x3;
    const float dotBetaIn = x3 * innerSegData.x1 + y3 * innerSegData.y1 - innerSegData.rt1 * innerSegData.rt1;
    const float r2 = crossBetaIn * crossBetaIn + dotBetaIn * dotBetaIn;
    const float sinBetaInCut = alpaka::math::sin(acc, betaInCut);
    const float sinBetaInCutSq = sinBetaInCut * sinBetaInCut;
    // A triplet admitted only by the widened bound is flagged and used only in quintuplets.
    constexpr float kT3PointingWiden = 1.7f;
    const float sinWideCut = alpaka::math::sin(acc, kT3PointingWiden * betaInCut);
    const float sinWideCutSq = sinWideCut * sinWideCut;

    float sinBetaInSq;
    bool cosPositive;
    if (innerSegData.innerSubdet == Endcap and innerSegData.middleSubdet == Endcap and outerSubdet == Endcap) {
      // EEE: check both alpha variants, use the one with smaller |betaIn|
      const float sinBetaInMin = innerSegData.sin_alphaRHmin * dotBetaIn - innerSegData.cos_alphaRHmin * crossBetaIn;
      const float sinBetaInMax = innerSegData.sin_alphaRHmax * dotBetaIn - innerSegData.cos_alphaRHmax * crossBetaIn;
      const float sqMin = sinBetaInMin * sinBetaInMin;
      const float sqMax = sinBetaInMax * sinBetaInMax;
      if (sqMin <= sqMax) {
        sinBetaInSq = sqMin;
        cosPositive = innerSegData.cos_alphaRHmin * dotBetaIn + innerSegData.sin_alphaRHmin * crossBetaIn > 0.f;
      } else {
        sinBetaInSq = sqMax;
        cosPositive = innerSegData.cos_alphaRHmax * dotBetaIn + innerSegData.sin_alphaRHmax * crossBetaIn > 0.f;
      }
    } else {
      const float sinBetaIn = innerSegData.sin_alpha * dotBetaIn - innerSegData.cos_alpha * crossBetaIn;
      sinBetaInSq = sinBetaIn * sinBetaIn;
      cosPositive = innerSegData.cos_alpha * dotBetaIn + innerSegData.sin_alpha * crossBetaIn > 0.f;
    }
    if (not cosPositive or sinBetaInSq >= sinWideCutSq * r2)
      return 0;
    return (sinBetaInSq < sinBetaInCutSq * r2) ? 1 : 2;
  }

  // The r-z cut of runTripletConstraintsAndAlgo (loosened by cutScale > 1 for the counting kernel).
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passTripletRZCountCut(TAcc const& acc,
                                                            ModulesConst modules,
                                                            MiniDoubletsConst mds,
                                                            SegmentsConst segments,
                                                            uint16_t innerInnerLowerModuleIndex,
                                                            uint16_t middleLowerModuleIndex,
                                                            uint16_t outerOuterLowerModuleIndex,
                                                            unsigned int innerSegmentIndex,
                                                            unsigned int outerSegmentIndex,
                                                            const float cutScale) {
    const unsigned int firstMDIndex = segments.mdIndices()[innerSegmentIndex][0];
    const unsigned int secondMDIndex = segments.mdIndices()[outerSegmentIndex][0];
    const unsigned int thirdMDIndex = segments.mdIndices()[outerSegmentIndex][1];
    T3HitCoords hitCoords;
    hitCoords.x1 = mds.anchorX()[firstMDIndex];
    hitCoords.y1 = mds.anchorY()[firstMDIndex];
    hitCoords.z1 = mds.anchorZ()[firstMDIndex];
    hitCoords.rt1 = mds.anchorRt()[firstMDIndex];
    hitCoords.x2 = mds.anchorX()[secondMDIndex];
    hitCoords.y2 = mds.anchorY()[secondMDIndex];
    hitCoords.z2 = mds.anchorZ()[secondMDIndex];
    hitCoords.rt2 = mds.anchorRt()[secondMDIndex];
    hitCoords.x3 = mds.anchorX()[thirdMDIndex];
    hitCoords.y3 = mds.anchorY()[thirdMDIndex];
    hitCoords.z3 = mds.anchorZ()[thirdMDIndex];
    hitCoords.rt3 = mds.anchorRt()[thirdMDIndex];
    const auto [circleRadius, circleCenterX, circleCenterY] = computeRadiusFromThreeAnchorHits(
        acc, hitCoords.x1, hitCoords.y1, hitCoords.x2, hitCoords.y2, hitCoords.x3, hitCoords.y3);
    short charge;
    return passRZConstraint(acc,
                            modules,
                            innerInnerLowerModuleIndex,
                            middleLowerModuleIndex,
                            outerOuterLowerModuleIndex,
                            hitCoords,
                            circleRadius,
                            circleCenterX,
                            circleCenterY,
                            charge,
                            cutScale);
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runTripletConstraintsAndAlgo(TAcc const& acc,
                                                                   MiniDoubletsConst mds,
                                                                   SegmentsConst segments,
                                                                   unsigned int innerSegmentIndex,
                                                                   unsigned int outerSegmentIndex,
                                                                   float& betaIn,
                                                                   float& betaInCut,
                                                                   float& circleRadius,
                                                                   float& circleCenterX,
                                                                   float& circleCenterY,
                                                                   const float ptCut,
                                                                   float (&t3Scores)[dnn::t3dnn::kOutputFeatures],
                                                                   short& charge) {
    const unsigned int firstMDIndex = segments.mdIndices()[innerSegmentIndex][0];
    const unsigned int secondMDIndex = segments.mdIndices()[outerSegmentIndex][0];
    const unsigned int thirdMDIndex = segments.mdIndices()[outerSegmentIndex][1];

    T3HitCoords hitCoords;
    hitCoords.x1 = mds.anchorX()[firstMDIndex];
    hitCoords.y1 = mds.anchorY()[firstMDIndex];
    hitCoords.z1 = mds.anchorZ()[firstMDIndex];
    hitCoords.rt1 = mds.anchorRt()[firstMDIndex];
    hitCoords.x2 = mds.anchorX()[secondMDIndex];
    hitCoords.y2 = mds.anchorY()[secondMDIndex];
    hitCoords.z2 = mds.anchorZ()[secondMDIndex];
    hitCoords.rt2 = mds.anchorRt()[secondMDIndex];
    hitCoords.x3 = mds.anchorX()[thirdMDIndex];
    hitCoords.y3 = mds.anchorY()[thirdMDIndex];
    hitCoords.z3 = mds.anchorZ()[thirdMDIndex];
    hitCoords.rt3 = mds.anchorRt()[thirdMDIndex];

    std::tie(circleRadius, circleCenterX, circleCenterY) = computeRadiusFromThreeAnchorHits(
        acc, hitCoords.x1, hitCoords.y1, hitCoords.x2, hitCoords.y2, hitCoords.x3, hitCoords.y3);

    // The r-z cut (passRZConstraint) is applied before, in step 1 of CreateTriplets.
    charge = tripletCharge(hitCoords);

    const float sdIn_alpha = __H2F(segments.dPhiChanges()[innerSegmentIndex]);

    const float drt_InSeg = hitCoords.rt2 - hitCoords.rt1;
    const float drt_tl_axis = alpaka::math::sqrt(acc,
                                                 (hitCoords.x3 - hitCoords.x1) * (hitCoords.x3 - hitCoords.x1) +
                                                     (hitCoords.y3 - hitCoords.y1) * (hitCoords.y3 - hitCoords.y1));

    //innerOuterAnchor - innerInnerAnchor
    const float rt_InSeg = alpaka::math::sqrt(acc,
                                              (hitCoords.x2 - hitCoords.x1) * (hitCoords.x2 - hitCoords.x1) +
                                                  (hitCoords.y2 - hitCoords.y1) * (hitCoords.y2 - hitCoords.y1));

    betaIn = sdIn_alpha - cms::alpakatools::reducePhiRange(
                              acc,
                              cms::alpakatools::phi(acc, hitCoords.x3 - hitCoords.x1, hitCoords.y3 - hitCoords.y1) -
                                  mds.anchorPhi()[firstMDIndex]);
    betaInCut =
        alpaka::math::asin(acc, alpaka::math::min(acc, (-rt_InSeg + drt_tl_axis) * k2Rinv1GeVf / ptCut, kSinAlphaMax)) +
        (0.02f / drt_InSeg);

    bool inference =
        lst::t3dnn::runInference(acc, mds, firstMDIndex, secondMDIndex, thirdMDIndex, circleRadius, betaIn, t3Scores);
    if (!inference)  // T3-building cut
      return false;

    return true;
  }

  // Segments by inner MD, one block per lower module: those starting at MD m are indexSegByMD[offsetSegByMD[m] + k],
  // k < nSegByMD[m] (zeroed before), in ascending segment index on the serial backend. Optional: the triplet counts
  // list only the segments that start a triplet (outer LS1 candidates of a T5), segInnerLowerModule gets their module.
  struct FillSegmentsByInnerMD {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  MiniDoubletsOccupancyConst mdOccupancy,
                                  SegmentsConst segments,
                                  SegmentsOccupancyConst segmentsOccupancy,
                                  ObjectRangesConst ranges,
                                  const unsigned int* __restrict__ nTripletsByModule,
                                  const uint32_t* __restrict__ nTripletsBySegment,
                                  uint32_t* __restrict__ nSegByMD,
                                  uint32_t* __restrict__ offsetSegByMD,
                                  uint32_t* __restrict__ indexSegByMD,
                                  uint16_t* __restrict__ segInnerLowerModule) const {
      for (auto module : cms::alpakatools::independent_groups(acc, modules.nLowerModules())) {
        const unsigned int nSegs = segmentsOccupancy.nSegments()[module];
        if (nSegs == 0 || (nTripletsByModule != nullptr && nTripletsByModule[module] == 0))
          continue;
        const unsigned int firstSeg = ranges.segmentModuleIndices()[module];
        for (auto k : cms::alpakatools::independent_group_elements(acc, nSegs)) {
          const unsigned int seg = firstSeg + k;
          if (nTripletsBySegment == nullptr || nTripletsBySegment[seg] > 0)
            alpaka::atomicAdd(acc, &nSegByMD[segments.mdIndices()[seg][0]], 1u, alpaka::hierarchy::Threads{});
        }
        alpaka::syncBlockThreads(acc);
        if (cms::alpakatools::once_per_block(acc)) {
          unsigned int offset = firstSeg;
          const unsigned int firstMD = ranges.mdRanges()[module][0];
          const unsigned int nMDs = mdOccupancy.nMDs()[module];
          for (unsigned int md = firstMD; md < firstMD + nMDs; ++md) {
            offsetSegByMD[md] = offset;
            offset += nSegByMD[md];
            nSegByMD[md] = 0;
          }
        }
        alpaka::syncBlockThreads(acc);
        for (auto k : cms::alpakatools::independent_group_elements(acc, nSegs)) {
          const unsigned int seg = firstSeg + k;
          if (nTripletsBySegment != nullptr && nTripletsBySegment[seg] == 0)
            continue;
          const unsigned int md = segments.mdIndices()[seg][0];
          const unsigned int position = alpaka::atomicAdd(acc, &nSegByMD[md], 1u, alpaka::hierarchy::Threads{});
          indexSegByMD[offsetSegByMD[md] + position] = seg;
          if (segInnerLowerModule != nullptr)
            segInnerLowerModule[seg] = module;
        }
        alpaka::syncBlockThreads(acc);
      }
    }
  };

  // Full T3 selection of one segment pair; a passing pair gets the next slot of the inner module's range and is
  // counted per segment and per inner MD (callers run several blocks per module, hence the device-wide atomics).
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void tryAddTriplet(TAcc const& acc,
                                                    MiniDoubletsConst mds,
                                                    SegmentsConst segments,
                                                    TripletsBuild triplets,
                                                    TripletsOccupancy tripletsOccupancy,
                                                    TripletsRanges tripletsRangesBySegment,
                                                    TripletCounts tripletCountsByMD,
                                                    ObjectRanges ranges,
                                                    const float ptCut,
                                                    unsigned int innerSegmentIndex,
                                                    unsigned int outerSegmentIndex,
                                                    uint16_t innerInnerLowerModuleIndex,
                                                    bool loosePointing) {
    float betaIn, betaInCut, circleRadius, circleCenterX, circleCenterY;
    short charge;
    float t3Scores[dnn::t3dnn::kOutputFeatures] = {0.f};

    bool success = runTripletConstraintsAndAlgo(acc,
                                                mds,
                                                segments,
                                                innerSegmentIndex,
                                                outerSegmentIndex,
                                                betaIn,
                                                betaInCut,
                                                circleRadius,
                                                circleCenterX,
                                                circleCenterY,
                                                ptCut,
                                                t3Scores,
                                                charge);
    if (!success)
      return;
    unsigned int totOccupancyTriplets = alpaka::atomicAdd(
        acc, &tripletsOccupancy.totOccupancyTriplets()[innerInnerLowerModuleIndex], 1u, alpaka::hierarchy::Blocks{});
    if (static_cast<int>(totOccupancyTriplets) >= ranges.tripletModuleOccupancy()[innerInnerLowerModuleIndex]) {
      alpaka::atomicAdd(acc, &ranges.nTripletOverflows(), 1u, alpaka::hierarchy::Blocks{});
#ifdef WARNINGS
      printf("Triplet excess alert! Module index = %d, Occupancy = %d\n",
             innerInnerLowerModuleIndex,
             totOccupancyTriplets);
#endif
      return;
    }
    const unsigned int tripletModuleIndex = alpaka::atomicAdd(
        acc, &tripletsOccupancy.nTriplets()[innerInnerLowerModuleIndex], 1u, alpaka::hierarchy::Blocks{});
    const unsigned int tripletIndex = ranges.tripletModuleIndices()[innerInnerLowerModuleIndex] + tripletModuleIndex;
    // Only count here: CompactTriplets fills the by-segment lists; the by-MD counts are read by the DNN features.
    alpaka::atomicAdd(acc, &tripletsRangesBySegment.n()[innerSegmentIndex], 1u, alpaka::hierarchy::Blocks{});
    auto const innerMDIndex = segments.mdIndices()[innerSegmentIndex][0];
    alpaka::atomicAdd(acc, &tripletCountsByMD.n()[innerMDIndex], 1u, alpaka::hierarchy::Blocks{});

    const uint8_t flags = loosePointing ? kT3LoosePointing : 0;
    addTripletToMemory(triplets,
                       innerSegmentIndex,
                       outerSegmentIndex,
                       betaIn,
                       betaInCut,
                       circleRadius,
                       circleCenterX,
                       circleCenterY,
                       tripletIndex,
                       t3Scores,
                       charge,
                       flags);
  }

  using T3PassMaskType = uint32_t;
  // Max length of outer LS pass bits CountSegmentConnections stores per inner segment (pass and loose-pointing masks)
  constexpr unsigned int kT3PassMaskBits = std::numeric_limits<T3PassMaskType>::digits;

  // Step 1 of the triplet creation, flat over the OT segments: the pairs passing the pointing and r-z cuts are stored
  // per inner module in the creation buffer (in segment order on the serial backends, arrival order on GPU).
  struct CreateTriplets {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  ModulesConst modules,
                                  MiniDoubletsConst mds,
                                  SegmentsT3CountsConst segT3Counts,
                                  SegmentsConst segments,
                                  TripletsScratch scratch,
                                  ObjectRanges ranges,
                                  const float ptCut,
                                  const uint32_t* __restrict__ nSegByMD,
                                  const uint32_t* __restrict__ offsetSegByMD,
                                  const uint32_t* __restrict__ indexSegByMD,
                                  const uint16_t* __restrict__ segInnerLowerModule,
                                  const unsigned int nSegments,
                                  uint32_t* __restrict__ matchCount,  // per-module cursor of the stored pairs
                                  const T3PassMaskType* __restrict__ t3PassMask,
                                  const T3PassMaskType* __restrict__ t3LoosePointingMask) const {
      for (unsigned int innerSegmentIndex : cms::alpakatools::uniform_elements_y(acc, nSegments)) {
        if (segT3Counts.connectedMax()[innerSegmentIndex] == 0)
          continue;

        const uint16_t innerInnerLowerModuleIndex = segInnerLowerModule[innerSegmentIndex];
        const uint16_t middleLowerModuleIndex = segments.outerLowerModuleIndices()[innerSegmentIndex];
        const unsigned int middleMDIndex = segments.mdIndices()[innerSegmentIndex][1];

        // Only the segments that start at the shared MD (in segment index order on the serial backends).
        const unsigned int nOuterSegments = nSegByMD[middleMDIndex];
        if (nOuterSegments == 0)
          continue;
        const unsigned int outerListOffset = offsetSegByMD[middleMDIndex];

        // Needed only by the candidates the count did not decide.
        T3InnerSegData innerSegData{};
        if (nOuterSegments > kT3PassMaskBits)
          innerSegData = loadT3InnerSegData(
              acc, mds, segments, modules, innerSegmentIndex, innerInnerLowerModuleIndex, middleLowerModuleIndex);
        const T3PassMaskType passBits = t3PassMask[innerSegmentIndex];
        const T3PassMaskType loosePointingBits = t3LoosePointingMask[innerSegmentIndex];

        for (unsigned int outerListIndex : cms::alpakatools::uniform_elements_x(acc, nOuterSegments)) {
          const unsigned int outerSegmentIndex = indexSegByMD[outerListOffset + outerListIndex];

          uint16_t outerOuterLowerModuleIndex = segments.outerLowerModuleIndices()[outerSegmentIndex];
          bool loosePointing;
          if (outerListIndex < kT3PassMaskBits) {
            if (!((passBits >> outerListIndex) & 1u))
              continue;
            loosePointing = (loosePointingBits >> outerListIndex) & 1u;
          } else {
            unsigned int thirdMDIndex = segments.mdIndices()[outerSegmentIndex][1];
            float x3 = mds.anchorX()[thirdMDIndex];
            float y3 = mds.anchorY()[thirdMDIndex];
            short outerSubdet = modules.subdets()[outerOuterLowerModuleIndex];

            const int pointing = passPointingConstraint(acc, innerSegData, x3, y3, outerSubdet, ptCut);
            if (not pointing)
              continue;
            loosePointing = (pointing == 2);
            // The r-z cut runs here instead of in step 2, so that step 1 keeps only what the counting kernel counted.
            if (!passTripletRZCountCut(acc,
                                       modules,
                                       mds,
                                       segments,
                                       innerInnerLowerModuleIndex,
                                       middleLowerModuleIndex,
                                       outerOuterLowerModuleIndex,
                                       innerSegmentIndex,
                                       outerSegmentIndex,
                                       1.f))
              continue;
          }

          // Match inner Sg and Outer Sg
          const unsigned int mIdx =
              alpaka::atomicAdd(acc, &matchCount[innerInnerLowerModuleIndex], 1u, alpaka::hierarchy::Blocks{});
          if (static_cast<int>(mIdx) >= ranges.tripletModuleOccupancy()[innerInnerLowerModuleIndex]) {
            alpaka::atomicAdd(acc, &ranges.nTripletOverflows(), 1u, alpaka::hierarchy::Blocks{});
            continue;
          }

          const unsigned int tripletIndex = ranges.tripletModuleIndices()[innerInnerLowerModuleIndex] + mIdx;
          scratch.segmentIndices()[tripletIndex][0] = innerSegmentIndex;
          scratch.segmentIndices()[tripletIndex][1] = outerSegmentIndex;
          scratch.flags()[tripletIndex] = loosePointing ? kT3LoosePointing : 0;
        }
      }
    }
  };

  // Flag of the creation-buffer slots that hold no match (memset before CreateTriplets).
  constexpr uint8_t kT3EmptySlot = 0xFF;

  // Step 2 of the triplet creation: the full selection of every match stored by CreateTriplets, flat over the
  // creation buffer (unused slots keep flags == kT3EmptySlot). Slot order = the per-module order of step 1.
  struct CreateTripletsFromMatches {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  MiniDoubletsConst mds,
                                  SegmentsConst segments,
                                  TripletsBuild triplets,
                                  TripletsOccupancy tripletsOccupancy,
                                  TripletsScratch scratch,
                                  TripletsRanges tripletsRangesBySegment,
                                  TripletCounts tripletCountsByMD,
                                  ObjectRanges ranges,
                                  const uint16_t* __restrict__ segInnerLowerModule,
                                  const unsigned int nSlots,
                                  const float ptCut) const {
      for (unsigned int slot : cms::alpakatools::uniform_elements(acc, nSlots)) {
        const uint8_t flags = scratch.flags()[slot];
        if (flags == kT3EmptySlot)
          continue;
        const unsigned int innerSegmentIndex = scratch.segmentIndices()[slot][0];
        const unsigned int outerSegmentIndex = scratch.segmentIndices()[slot][1];
        tryAddTriplet(acc,
                      mds,
                      segments,
                      triplets,
                      tripletsOccupancy,
                      tripletsRangesBySegment,
                      tripletCountsByMD,
                      ranges,
                      ptCut,
                      innerSegmentIndex,
                      outerSegmentIndex,
                      segInnerLowerModule[innerSegmentIndex],
                      flags & kT3LoosePointing);
      }
    }
  };

  struct CountSegmentConnections {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  ModulesConst modules,
                                  MiniDoubletsConst mds,
                                  SegmentsT3Counts segT3Counts,
                                  Segments segments,
                                  const float ptCut,
                                  const uint32_t* __restrict__ nSegByMD,
                                  const uint32_t* __restrict__ offsetSegByMD,
                                  const uint32_t* __restrict__ indexSegByMD,
                                  const uint16_t* __restrict__ segInnerLowerModule,
                                  const unsigned int nSegments,
                                  T3PassMaskType* __restrict__ t3PassMask,
                                  T3PassMaskType* __restrict__ t3LoosePointingMask) const {
      // Flat over the OT segments, each one handled by the threads of one block: the atomicAdd below with
      // hierarchy::Threads{} requires one block in x.
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[1] == 1));
      const auto& mdIndices = segments.mdIndices();
      const auto& outerLowerModuleIndices = segments.outerLowerModuleIndices();

      for (unsigned int innerSegmentIndex : cms::alpakatools::uniform_elements_y(acc, nSegments)) {
        const uint16_t innerInnerLowerModuleIndex = segInnerLowerModule[innerSegmentIndex];
        const uint16_t middleLowerModuleIndex = outerLowerModuleIndices[innerSegmentIndex];
        const unsigned int middleMDIndex = mdIndices[innerSegmentIndex][1];

        // Only the segments that start at the shared MD.
        const unsigned int nOuterSegments = nSegByMD[middleMDIndex];
        if (nOuterSegments == 0)
          continue;
        const unsigned int outerListOffset = offsetSegByMD[middleMDIndex];

        T3InnerSegData innerSegData = loadT3InnerSegData(
            acc, mds, segments, modules, innerSegmentIndex, innerInnerLowerModuleIndex, middleLowerModuleIndex);

        for (unsigned int outerListIndex : cms::alpakatools::uniform_elements_x(acc, nOuterSegments)) {
          const unsigned int outerSegmentIndex = indexSegByMD[outerListOffset + outerListIndex];

          unsigned int thirdMDIndex = mdIndices[outerSegmentIndex][1];
          uint16_t outerOuterLowerModuleIndex = outerLowerModuleIndices[outerSegmentIndex];
          float x3 = mds.anchorX()[thirdMDIndex];
          float y3 = mds.anchorY()[thirdMDIndex];
          short outerSubdet = modules.subdets()[outerOuterLowerModuleIndex];

          const int pointing = passPointingConstraint(acc, innerSegData, x3, y3, outerSubdet, ptCut);
          if (not pointing)
            continue;

          // Candidates in the pass mask: the exact step-1 decision, stored; the others: a superset of step 1.
          const bool inPassMask = outerListIndex < kT3PassMaskBits;
          const bool passRZ = passTripletRZCountCut(acc,
                                                    modules,
                                                    mds,
                                                    segments,
                                                    innerInnerLowerModuleIndex,
                                                    middleLowerModuleIndex,
                                                    outerOuterLowerModuleIndex,
                                                    innerSegmentIndex,
                                                    outerSegmentIndex,
                                                    inPassMask ? 1.f : kCountCutSlack);
          if (passRZ && inPassMask) {
            alpaka::atomicOr(
                acc, &t3PassMask[innerSegmentIndex], T3PassMaskType(1) << outerListIndex, alpaka::hierarchy::Threads{});
            if (pointing == 2)
              alpaka::atomicOr(acc,
                               &t3LoosePointingMask[innerSegmentIndex],
                               T3PassMaskType(1) << outerListIndex,
                               alpaka::hierarchy::Threads{});
          }
          if (passRZ) {
            alpaka::atomicAdd(acc, &segT3Counts.connectedMax()[innerSegmentIndex], 1u, alpaka::hierarchy::Threads{});
          }
        }
      }
    }
  };

  struct CreateTripletArrayRanges {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  ObjectRanges ranges,
                                  SegmentsT3CountsConst segT3Counts,
                                  SegmentsOccupancyConst segOcc) const {
      // 1-block kernel
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      int& nTotalTriplets = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      if (cms::alpakatools::once_per_block(acc))
        nTotalTriplets = 0;
      alpaka::syncBlockThreads(acc);

      for (uint16_t innerLowerModuleArrayIdx : cms::alpakatools::uniform_elements(acc, modules.nLowerModules())) {
        const unsigned int nInnerSegments = segOcc.nSegments()[innerLowerModuleArrayIdx];
        if (nInnerSegments == 0) {
          ranges.tripletModuleIndices()[innerLowerModuleArrayIdx] = nTotalTriplets;
          ranges.tripletModuleOccupancy()[innerLowerModuleArrayIdx] = 0;
          continue;
        }

        // Sum the connected counts of all segments in this module.
        const unsigned int firstSegIdx = ranges.segmentRanges()[innerLowerModuleArrayIdx][0];
        int dynamicCount = 0;
        for (unsigned int s = 0; s < nInnerSegments; ++s) {
          dynamicCount += segT3Counts.connectedMax()[firstSegIdx + s];
        }

        ranges.tripletModuleOccupancy()[innerLowerModuleArrayIdx] = dynamicCount;
        unsigned int nTotT = alpaka::atomicAdd(acc, &nTotalTriplets, dynamicCount, alpaka::hierarchy::Threads{});
        ranges.tripletModuleIndices()[innerLowerModuleArrayIdx] = nTotT;
      }

      // Wait for all threads to finish before reporting final values
      alpaka::syncBlockThreads(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        ranges.nTotalTrips() = nTotalTriplets;
        ranges.nTripletOverflows() = 0;
      }
    }
  };

  // Lays the created triplets out densely, module by module: module m gets [compact offset, + nTriplets[m]), with
  // the offsets from CompactModuleOffsetsKernel. Saves the creation-time module offsets in looseModuleIndices for
  // CompactTriplets.
  struct SetCompactTripletModuleIndices {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  TripletsOccupancyConst tripletsOccupancy,
                                  const int* compactModuleIndices,
                                  ObjectRanges ranges,
                                  int* looseModuleIndices) const {
      for (uint16_t i : cms::alpakatools::uniform_elements(acc, modules.nLowerModules())) {
        looseModuleIndices[i] = ranges.tripletModuleIndices()[i];
        ranges.tripletModuleIndices()[i] = compactModuleIndices[i];
        ranges.tripletModuleOccupancy()[i] = tripletsOccupancy.nTriplets()[i];
      }
      if (cms::alpakatools::once_per_grid(acc))
        ranges.nTotalTrips() = compactModuleIndices[modules.nLowerModules()];
    }
  };

  // Exclusive prefix sum of n()[first, first + count) from base into offset() by the threads of the block, one
  // contiguous chunk each (chunkSums: one entry per thread); n() is reset as the fill cursor.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void setTripletRangeOffsets(TAcc const& acc,
                                                             TripletsRanges tripletsRanges,
                                                             unsigned int first,
                                                             unsigned int count,
                                                             unsigned int base,
                                                             unsigned int flatThreadIdx,
                                                             unsigned int flatThreadExtent,
                                                             unsigned int* chunkSums) {
    const unsigned int chunk = cms::alpakatools::divide_up_by(count, flatThreadExtent);
    const unsigned int begin = cms::alpakatools::idx_min(flatThreadIdx * chunk, count);
    const unsigned int end = cms::alpakatools::idx_min(begin + chunk, count);
    unsigned int sum = 0;
    for (unsigned int i = begin; i < end; ++i)
      sum += tripletsRanges.n()[first + i];
    chunkSums[flatThreadIdx] = sum;
    alpaka::syncBlockThreads(acc);
    if (cms::alpakatools::once_per_block(acc)) {
      unsigned int total = base;
      // Only the threads with a non-empty chunk.
      const unsigned int nUsedChunks = chunk == 0 ? 0 : cms::alpakatools::divide_up_by(count, chunk);
      for (unsigned int chunkIndex = 0; chunkIndex < nUsedChunks; ++chunkIndex) {
        const unsigned int chunkSum = chunkSums[chunkIndex];
        chunkSums[chunkIndex] = total;
        total += chunkSum;
      }
    }
    alpaka::syncBlockThreads(acc);
    unsigned int offset = chunkSums[flatThreadIdx];
    for (unsigned int i = begin; i < end; ++i) {
      tripletsRanges.offset()[first + i] = offset;
      offset += tripletsRanges.n()[first + i];
      tripletsRanges.n()[first + i] = 0;
    }
    alpaka::syncBlockThreads(acc);
  }

  // Copies the triplets of each module from the creation buffer into the exact-size one (initializing the
  // columns filled by later stages) and fills the by-segment lists, whose counts come from creation.
  struct CompactTriplets {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  SegmentsConst segments,
                                  SegmentsOccupancyConst segmentsOccupancy,
                                  TripletsBuildConst looseTriplets,
                                  TripletsOccupancyConst tripletsOccupancy,
                                  Triplets triplets,
                                  TripletsBySegment tripletsBySegment,
                                  TripletsRanges tripletsRangesBySegment,
                                  ObjectRangesConst ranges,
                                  const int* looseModuleIndices,
                                  const uint16_t* index_gpu,
                                  uint16_t nonZeroModules) const {
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[1] == 1) &&
                        (alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[2] == 1));

      const auto threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
      const auto blockDim = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
      const int flatThreadIdxXY = threadIdx.y() * blockDim.x() + threadIdx.x();
      const int flatThreadExtent = blockDim.x() * blockDim.y();

      constexpr int kMaxThreads = 1024;
      ALPAKA_ASSERT_ACC(flatThreadExtent <= kMaxThreads);
      auto& chunkSums = alpaka::declareSharedVar<unsigned int[kMaxThreads], __COUNTER__>(acc);

      for (uint16_t innerLowerModuleArrayIdx : cms::alpakatools::uniform_groups_z(acc, nonZeroModules)) {
        const uint16_t lowerModuleIndex = index_gpu[innerLowerModuleArrayIdx];
        const unsigned int compactOffset = ranges.tripletModuleIndices()[lowerModuleIndex];

        setTripletRangeOffsets(acc,
                               tripletsRangesBySegment,
                               ranges.segmentRanges()[lowerModuleIndex][0],
                               segmentsOccupancy.nSegments()[lowerModuleIndex],
                               compactOffset,
                               flatThreadIdxXY,
                               flatThreadExtent,
                               chunkSums);

        const unsigned int looseOffset = looseModuleIndices[lowerModuleIndex];
        const unsigned int nTriplets = tripletsOccupancy.nTriplets()[lowerModuleIndex];
        for (unsigned int i = flatThreadIdxXY; i < nTriplets; i += flatThreadExtent) {
          const unsigned int src = looseOffset + i;
          const unsigned int dst = compactOffset + i;
          const unsigned int innerSegmentIndex = looseTriplets.segmentIndices()[src][0];
          const unsigned int outerSegmentIndex = looseTriplets.segmentIndices()[src][1];
          triplets.segmentIndices()[dst][0] = innerSegmentIndex;
          triplets.segmentIndices()[dst][1] = outerSegmentIndex;
          triplets.lowerModuleIndices()[dst][0] = lowerModuleIndex;
          triplets.lowerModuleIndices()[dst][1] = segments.outerLowerModuleIndices()[innerSegmentIndex];
          triplets.lowerModuleIndices()[dst][2] = segments.outerLowerModuleIndices()[outerSegmentIndex];
          triplets.centerX()[dst] = looseTriplets.centerX()[src];
          triplets.centerY()[dst] = looseTriplets.centerY()[src];
          triplets.radius()[dst] = looseTriplets.radius()[src];
          triplets.fakeScore()[dst] = looseTriplets.fakeScore()[src];
          triplets.promptScore()[dst] = looseTriplets.promptScore()[src];
          triplets.displacedScore()[dst] = looseTriplets.displacedScore()[src];
          triplets.charge()[dst] = looseTriplets.charge()[src];
          triplets.flags()[dst] = looseTriplets.flags()[src];
          triplets.partOfPT5()[dst] = false;
          triplets.partOfT5()[dst] = false;
          triplets.partOfPT3()[dst] = false;
#ifdef CUT_VALUE_DEBUG
          triplets.betaIn()[dst] = looseTriplets.betaIn()[src];
          triplets.betaInCut()[dst] = looseTriplets.betaInCut()[src];
#endif

          const unsigned int bySegmentIndex =
              tripletsRangesBySegment.offset()[innerSegmentIndex] +
              alpaka::atomicAdd(acc, &tripletsRangesBySegment.n()[innerSegmentIndex], 1u, alpaka::hierarchy::Threads{});
          tripletsBySegment.tripletIndex()[bySegmentIndex] = dst;
        }
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
