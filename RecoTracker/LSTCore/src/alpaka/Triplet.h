#ifndef RecoTracker_LSTCore_src_alpaka_Triplet_h
#define RecoTracker_LSTCore_src_alpaka_Triplet_h

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"
#include "RecoTracker/LSTCore/interface/TripletsSoA.h"
#include "RecoTracker/LSTCore/interface/Circle.h"

#include "NeuralNetwork.h"

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
                                                         float betaIn,
                                                         float betaInCut,
                                                         float circleRadius,
                                                         float circleCenterX,
                                                         float circleCenterY,
                                                         unsigned int tripletIndex,
                                                         float (&t3Scores)[dnn::t3dnn::kOutputFeatures],
                                                         short charge) {
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

    triplets.charge()[tripletIndex] = charge;
#ifdef CUT_VALUE_DEBUG
    triplets.betaInCut()[tripletIndex] = betaInCut;
#endif

    triplets.fakeScore()[tripletIndex] = t3Scores[0];
    triplets.promptScore()[tripletIndex] = t3Scores[1];
    triplets.displacedScore()[tripletIndex] = t3Scores[2];
  }

  template <alpaka::concepts::Acc TAcc>
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
                                                       float circleCenterY,
                                                       short& charge) {
    // Using lst_layer numbering convention defined in ModuleMethods.h
    const short layer1 = modules.lstLayers()[innerInnerLowerModuleIndex];
    const short layer2 = modules.lstLayers()[middleLowerModuleIndex];
    const short layer3 = modules.lstLayers()[outerOuterLowerModuleIndex];

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

    //get the x,y position of each MD
    const float x1 = mds.anchorX()[firstMDIndex] / 100;
    const float x2 = mds.anchorX()[secondMDIndex] / 100;
    const float x3 = mds.anchorX()[thirdMDIndex] / 100;

    const float y1 = mds.anchorY()[firstMDIndex] / 100;
    const float y2 = mds.anchorY()[secondMDIndex] / 100;
    const float y3 = mds.anchorY()[thirdMDIndex] / 100;

    float cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
    charge = -1 * ((int)copysignf(1.0f, cross));

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
      float sol1 = (-B + alpaka::math::sqrt(acc, B * B - 4 * A * C)) / (2 * A);
      float sol2 = (-B - alpaka::math::sqrt(acc, B * B - 4 * A * C)) / (2 * A);
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
      float x = x_init + px / a * alpaka::math::sin(acc, rou * s) - py / a * (1 - alpaka::math::cos(acc, rou * s));
      float y = y_init + py / a * alpaka::math::sin(acc, rou * s) + px / a * (1 - alpaka::math::cos(acc, rou * s));
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

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPointingConstraint(TAcc const& acc,
                                                             ModulesConst modules,
                                                             MiniDoubletsConst mds,
                                                             SegmentsConst segments,
                                                             unsigned int firstMDIndex,
                                                             unsigned int secondMDIndex,
                                                             unsigned int thirdMDIndex,
                                                             uint16_t innerInnerLowerModuleIndex,
                                                             uint16_t middleLowerModuleIndex,
                                                             uint16_t outerOuterLowerModuleIndex,
                                                             unsigned int innerSegmentIndex,
                                                             const float ptCut) {
    const float x1 = mds.anchorX()[firstMDIndex];
    const float x2 = mds.anchorX()[secondMDIndex];
    const float x3 = mds.anchorX()[thirdMDIndex];
    const float y1 = mds.anchorY()[firstMDIndex];
    const float y2 = mds.anchorY()[secondMDIndex];
    const float y3 = mds.anchorY()[thirdMDIndex];

    const short innerInnerLowerModuleSubdet = modules.subdets()[innerInnerLowerModuleIndex];
    const short middleLowerModuleSubdet = modules.subdets()[middleLowerModuleIndex];
    const short outerOuterLowerModuleSubdet = modules.subdets()[outerOuterLowerModuleIndex];

    const float rt_InLo = mds.anchorRt()[firstMDIndex];
    const float rt_InOut = mds.anchorRt()[secondMDIndex];
    const float sdIn_alpha = __H2F(segments.dPhiChanges()[innerSegmentIndex]);

    const float drt_InSeg = rt_InOut - rt_InLo;
    const float drt_tl_axis = alpaka::math::sqrt(acc, (x3 - x1) * (x3 - x1) + (y3 - y1) * (y3 - y1));

    //innerOuterAnchor - innerInnerAnchor
    const float rt_InSeg = alpaka::math::sqrt(acc, (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));

    const float betaIn =
        sdIn_alpha - cms::alpakatools::reducePhiRange(
                         acc, cms::alpakatools::phi(acc, x3 - x1, y3 - y1) - mds.anchorPhi()[firstMDIndex]);
    const float betaInCut =
        alpaka::math::asin(acc, alpaka::math::min(acc, (-rt_InSeg + drt_tl_axis) * k2Rinv1GeVf / ptCut, kSinAlphaMax)) +
        (0.02f / drt_InSeg);

    float betaInRHmin = betaIn;

    if (innerInnerLowerModuleSubdet == Endcap and middleLowerModuleSubdet == Endcap and
        outerOuterLowerModuleSubdet == Endcap) {
      float sdIn_alphaRHmin = __H2F(segments.dPhiChangeMins()[innerSegmentIndex]);
      float sdIn_alphaRHmax = __H2F(segments.dPhiChangeMaxs()[innerSegmentIndex]);

      betaInRHmin = betaIn + sdIn_alphaRHmin - sdIn_alpha;
      float betaInRHmax = betaIn + sdIn_alphaRHmax - sdIn_alpha;
      float swapTemp;

      if (alpaka::math::abs(acc, betaInRHmin) > alpaka::math::abs(acc, betaInRHmax)) {
        swapTemp = betaInRHmin;
        betaInRHmin = betaInRHmax;
        betaInRHmax = swapTemp;
      }
    }

    //Beta cut
    return alpaka::math::abs(acc, betaInRHmin) < betaInCut;
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runTripletConstraintsAndAlgo(TAcc const& acc,
                                                                   ModulesConst modules,
                                                                   MiniDoubletsConst mds,
                                                                   SegmentsConst segments,
                                                                   uint16_t innerInnerLowerModuleIndex,
                                                                   uint16_t middleLowerModuleIndex,
                                                                   uint16_t outerOuterLowerModuleIndex,
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

    const float x1 = mds.anchorX()[firstMDIndex];
    const float x2 = mds.anchorX()[secondMDIndex];
    const float x3 = mds.anchorX()[thirdMDIndex];
    const float y1 = mds.anchorY()[firstMDIndex];
    const float y2 = mds.anchorY()[secondMDIndex];
    const float y3 = mds.anchorY()[thirdMDIndex];

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
                             circleCenterY,
                             charge))
      return false;

    const float rt_InLo = mds.anchorRt()[firstMDIndex];
    const float rt_InOut = mds.anchorRt()[secondMDIndex];
    const float sdIn_alpha = __H2F(segments.dPhiChanges()[innerSegmentIndex]);

    const float drt_InSeg = rt_InOut - rt_InLo;
    const float drt_tl_axis = alpaka::math::sqrt(acc, (x3 - x1) * (x3 - x1) + (y3 - y1) * (y3 - y1));

    //innerOuterAnchor - innerInnerAnchor
    const float rt_InSeg = alpaka::math::sqrt(acc, (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));

    betaIn = sdIn_alpha - cms::alpakatools::reducePhiRange(
                              acc, cms::alpakatools::phi(acc, x3 - x1, y3 - y1) - mds.anchorPhi()[firstMDIndex]);
    betaInCut =
        alpaka::math::asin(acc, alpaka::math::min(acc, (-rt_InSeg + drt_tl_axis) * k2Rinv1GeVf / ptCut, kSinAlphaMax)) +
        (0.02f / drt_InSeg);

    bool inference =
        lst::t3dnn::runInference(acc, mds, firstMDIndex, secondMDIndex, thirdMDIndex, circleRadius, betaIn, t3Scores);
    if (!inference)  // T3-building cut
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
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[1] == 1) &&
                        (alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[2] == 1));

      int& matchCount =
          alpaka::declareSharedVar<int, __COUNTER__>(acc);  // AtomicAdd does not support uint16_t variable
      const auto threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
      const auto blockDim = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);

      const int threadIdX = threadIdx.x();
      const int threadIdY = threadIdx.y();
      const int blockSizeX = blockDim.x();
      const int blockSizeY = blockDim.y();
      const int blockSize = blockSizeX * blockSizeY;
      const int flatThreadIdxXY = threadIdY * blockSizeX + threadIdX;
      const int flatThreadExtent = blockSize;  // total threads per block

      for (uint16_t innerLowerModuleArrayIdx : cms::alpakatools::uniform_groups_z(acc, nonZeroModules)) {
        if (cms::alpakatools::once_per_block(acc)) {
          matchCount = 0;
        }

        uint16_t innerInnerLowerModuleIndex = index_gpu[innerLowerModuleArrayIdx];
        if (innerInnerLowerModuleIndex >= modules.nLowerModules())
          continue;

        uint16_t nConnectedModules = modules.nConnectedModules()[innerInnerLowerModuleIndex];
        if (nConnectedModules == 0)
          continue;

        unsigned int nInnerSegments = segmentsOccupancy.nSegments()[innerInnerLowerModuleIndex];

        if (nInnerSegments == 0)
          continue;

        alpaka::syncBlockThreads(acc);

        // Step 1: Make inner and outer SG pairs
        for (unsigned int innerSegmentArrayIndex : cms::alpakatools::uniform_elements_y(acc, nInnerSegments)) {
          unsigned int innerSegmentIndex =
              ranges.segmentRanges()[innerInnerLowerModuleIndex][0] + innerSegmentArrayIndex;

          uint16_t middleLowerModuleIndex = segments.outerLowerModuleIndices()[innerSegmentIndex];
          int middleMDIndiceInner = segments.mdIndices()[innerSegmentIndex][1];

          unsigned int nOuterSegments = segmentsOccupancy.nSegments()[middleLowerModuleIndex];
          for (unsigned int outerSegmentArrayIndex : cms::alpakatools::uniform_elements_x(acc, nOuterSegments)) {
            unsigned int outerSegmentIndex = ranges.segmentRanges()[middleLowerModuleIndex][0] + outerSegmentArrayIndex;

            int middleMDIndiceOuter = segments.mdIndices()[outerSegmentIndex][0];
            if (middleMDIndiceInner != middleMDIndiceOuter)
              continue;

            uint16_t outerOuterLowerModuleIndex = segments.outerLowerModuleIndices()[outerSegmentIndex];
            unsigned int firstMDIndex = segments.mdIndices()[innerSegmentIndex][0];
            unsigned int secondMDIndex = segments.mdIndices()[outerSegmentIndex][0];
            unsigned int thirdMDIndex = segments.mdIndices()[outerSegmentIndex][1];

            if (not passPointingConstraint(acc,
                                           modules,
                                           mds,
                                           segments,
                                           firstMDIndex,
                                           secondMDIndex,
                                           thirdMDIndex,
                                           innerInnerLowerModuleIndex,
                                           middleLowerModuleIndex,
                                           outerOuterLowerModuleIndex,
                                           innerSegmentIndex,
                                           ptCut))
              continue;

            // Match inner Sg and Outer Sg
            int mIdx = alpaka::atomicAdd(acc, &matchCount, 1, alpaka::hierarchy::Threads{});

            unsigned int tripletIndex = ranges.tripletModuleIndices()[innerInnerLowerModuleIndex] + mIdx;

#ifdef WARNINGS
            const unsigned int rightBound =
                static_cast<unsigned int>(ranges.tripletModuleIndices()[innerInnerLowerModuleIndex + 1]);
            if (tripletIndex >= rightBound) {
              printf(
                  "Triplet module occupancy alert! module triplet starting index  = %d, Pair triplet index = "
                  "%d, next module triplet starting index = %d\n",
                  ranges.tripletModuleIndices()[innerInnerLowerModuleIndex],
                  mIdx,
                  ranges.tripletModuleIndices()[innerInnerLowerModuleIndex + 1]);
            }
#endif

            triplets.preAllocatedSegmentIndices()[tripletIndex][0] = innerSegmentIndex;
            triplets.preAllocatedSegmentIndices()[tripletIndex][1] = outerSegmentIndex;
          }
        }

        alpaka::syncBlockThreads(acc);
        if (matchCount == 0) {
          continue;
        }

        // Step 2: Parallel processing of segment pairs
        for (int i = flatThreadIdxXY; i < matchCount; i += flatThreadExtent) {
          unsigned int tripletIndex = ranges.tripletModuleIndices()[innerInnerLowerModuleIndex] + i;
          unsigned int innerSegmentIndex = triplets.preAllocatedSegmentIndices()[tripletIndex][0];
          unsigned int outerSegmentIndex = triplets.preAllocatedSegmentIndices()[tripletIndex][1];

          uint16_t middleLowerModuleIndex = segments.outerLowerModuleIndices()[innerSegmentIndex];
          uint16_t outerOuterLowerModuleIndex = segments.outerLowerModuleIndices()[outerSegmentIndex];

          float betaIn, betaInCut, circleRadius, circleCenterX, circleCenterY;
          short charge;

          float t3Scores[dnn::t3dnn::kOutputFeatures] = {0.f};

          bool success = runTripletConstraintsAndAlgo(acc,
                                                      modules,
                                                      mds,
                                                      segments,
                                                      innerInnerLowerModuleIndex,
                                                      middleLowerModuleIndex,
                                                      outerOuterLowerModuleIndex,
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
          if (success) {
            unsigned int totOccupancyTriplets =
                alpaka::atomicAdd(acc,
                                  &tripletsOccupancy.totOccupancyTriplets()[innerInnerLowerModuleIndex],
                                  1u,
                                  alpaka::hierarchy::Threads{});
            if (static_cast<int>(totOccupancyTriplets) >= ranges.tripletModuleOccupancy()[innerInnerLowerModuleIndex]) {
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
                                 betaIn,
                                 betaInCut,
                                 circleRadius,
                                 circleCenterX,
                                 circleCenterY,
                                 tripletIndex,
                                 t3Scores,
                                 charge);
            }
          }
        }
      }
    }
  };

  struct CountSegmentConnections {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  ModulesConst modules,
                                  Segments segments,
                                  SegmentsOccupancyConst segOcc,
                                  ObjectRangesConst ranges) const {
      // The atomicAdd below with hierarchy::Threads{} requires one block in x, y dimensions.
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[1] == 1) &&
                        (alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[2] == 1));
      const auto& mdIndices = segments.mdIndices();
      const auto& outerLowerModuleIndices = segments.outerLowerModuleIndices();
      const auto& segmentRanges = ranges.segmentRanges();

      for (uint16_t innerLowerModuleArrayIdx : cms::alpakatools::uniform_groups_z(acc, modules.nLowerModules())) {
        const unsigned int nInnerSegments = segOcc.nSegments()[innerLowerModuleArrayIdx];
        if (nInnerSegments == 0)
          continue;

        for (unsigned int innerSegmentArrayIndex : cms::alpakatools::uniform_elements_y(acc, nInnerSegments)) {
          const unsigned int innerSegmentIndex = segmentRanges[innerLowerModuleArrayIdx][0] + innerSegmentArrayIndex;
          const uint16_t middleLowerModuleIndex = outerLowerModuleIndices[innerSegmentIndex];
          const unsigned int mdShared = mdIndices[innerSegmentIndex][1];

          const unsigned int nOuterSegments = segOcc.nSegments()[middleLowerModuleIndex];
          if (nOuterSegments == 0)
            continue;

          for (unsigned int outerSegmentArrayIndex : cms::alpakatools::uniform_elements_x(acc, nOuterSegments)) {
            const unsigned int outerSegmentIndex = segmentRanges[middleLowerModuleIndex][0] + outerSegmentArrayIndex;

            // Increment the count of connected segments for this segment.
            if (mdIndices[outerSegmentIndex][0] == mdShared) {
              alpaka::atomicAdd(acc, &segments.connectedMax()[innerSegmentIndex], 1u, alpaka::hierarchy::Threads{});
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
                                  SegmentsConst segments,
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
          dynamicCount += segments.connectedMax()[firstSegIdx + s];
        }

        ranges.tripletModuleOccupancy()[innerLowerModuleArrayIdx] = dynamicCount;
        unsigned int nTotT = alpaka::atomicAdd(acc, &nTotalTriplets, dynamicCount, alpaka::hierarchy::Threads{});
        ranges.tripletModuleIndices()[innerLowerModuleArrayIdx] = nTotT;
      }

      // Wait for all threads to finish before reporting final values
      alpaka::syncBlockThreads(acc);
      if (cms::alpakatools::once_per_block(acc))
        ranges.nTotalTrips() = nTotalTriplets;
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
