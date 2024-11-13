#ifndef RecoTracker_LSTCore_src_alpaka_PixelQuintuplet_h
#define RecoTracker_LSTCore_src_alpaka_PixelQuintuplet_h

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsSoA.h"
#include "RecoTracker/LSTCore/interface/PixelTripletsSoA.h"
#include "RecoTracker/LSTCore/interface/QuintupletsSoA.h"
#include "RecoTracker/LSTCore/interface/SegmentsSoA.h"
#include "RecoTracker/LSTCore/interface/TripletsSoA.h"

#include "Hit.h"
#include "PixelTriplet.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addPixelQuintupletToMemory(ModulesConst modules,
                                                                 MiniDoubletsConst mds,
                                                                 SegmentsConst segments,
                                                                 QuintupletsConst quintuplets,
                                                                 PixelQuintuplets pixelQuintuplets,
                                                                 unsigned int pixelIndex,
                                                                 unsigned int t5Index,
                                                                 unsigned int pixelQuintupletIndex,
                                                                 float rzChiSquared,
                                                                 float rPhiChiSquared,
                                                                 float rPhiChiSquaredInwards,
                                                                 float score,
                                                                 float eta,
                                                                 float phi,
                                                                 float pixelRadius,
                                                                 float quintupletRadius,
                                                                 float centerX,
                                                                 float centerY) {
    pixelQuintuplets.pixelSegmentIndices()[pixelQuintupletIndex] = pixelIndex;
    pixelQuintuplets.quintupletIndices()[pixelQuintupletIndex] = t5Index;
    pixelQuintuplets.isDup()[pixelQuintupletIndex] = false;
    pixelQuintuplets.score()[pixelQuintupletIndex] = __F2H(score);
    pixelQuintuplets.eta()[pixelQuintupletIndex] = __F2H(eta);
    pixelQuintuplets.phi()[pixelQuintupletIndex] = __F2H(phi);

    pixelQuintuplets.pixelRadius()[pixelQuintupletIndex] = __F2H(pixelRadius);
    pixelQuintuplets.quintupletRadius()[pixelQuintupletIndex] = __F2H(quintupletRadius);
    pixelQuintuplets.centerX()[pixelQuintupletIndex] = __F2H(centerX);
    pixelQuintuplets.centerY()[pixelQuintupletIndex] = __F2H(centerY);

    pixelQuintuplets.logicalLayers()[pixelQuintupletIndex][0] = 0;
    pixelQuintuplets.logicalLayers()[pixelQuintupletIndex][1] = 0;
    pixelQuintuplets.logicalLayers()[pixelQuintupletIndex][2] = quintuplets.logicalLayers()[t5Index][0];
    pixelQuintuplets.logicalLayers()[pixelQuintupletIndex][3] = quintuplets.logicalLayers()[t5Index][1];
    pixelQuintuplets.logicalLayers()[pixelQuintupletIndex][4] = quintuplets.logicalLayers()[t5Index][2];
    pixelQuintuplets.logicalLayers()[pixelQuintupletIndex][5] = quintuplets.logicalLayers()[t5Index][3];
    pixelQuintuplets.logicalLayers()[pixelQuintupletIndex][6] = quintuplets.logicalLayers()[t5Index][4];

    pixelQuintuplets.lowerModuleIndices()[pixelQuintupletIndex][0] = segments.innerLowerModuleIndices()[pixelIndex];
    pixelQuintuplets.lowerModuleIndices()[pixelQuintupletIndex][1] = segments.outerLowerModuleIndices()[pixelIndex];
    pixelQuintuplets.lowerModuleIndices()[pixelQuintupletIndex][2] = quintuplets.lowerModuleIndices()[t5Index][0];
    pixelQuintuplets.lowerModuleIndices()[pixelQuintupletIndex][3] = quintuplets.lowerModuleIndices()[t5Index][1];
    pixelQuintuplets.lowerModuleIndices()[pixelQuintupletIndex][4] = quintuplets.lowerModuleIndices()[t5Index][2];
    pixelQuintuplets.lowerModuleIndices()[pixelQuintupletIndex][5] = quintuplets.lowerModuleIndices()[t5Index][3];
    pixelQuintuplets.lowerModuleIndices()[pixelQuintupletIndex][6] = quintuplets.lowerModuleIndices()[t5Index][4];

    unsigned int pixelInnerMD = segments.mdIndices()[pixelIndex][0];
    unsigned int pixelOuterMD = segments.mdIndices()[pixelIndex][1];

    pixelQuintuplets.hitIndices()[pixelQuintupletIndex][0] = mds.anchorHitIndices()[pixelInnerMD];
    pixelQuintuplets.hitIndices()[pixelQuintupletIndex][1] = mds.outerHitIndices()[pixelInnerMD];
    pixelQuintuplets.hitIndices()[pixelQuintupletIndex][2] = mds.anchorHitIndices()[pixelOuterMD];
    pixelQuintuplets.hitIndices()[pixelQuintupletIndex][3] = mds.outerHitIndices()[pixelOuterMD];

    pixelQuintuplets.hitIndices()[pixelQuintupletIndex][4] = quintuplets.hitIndices()[t5Index][0];
    pixelQuintuplets.hitIndices()[pixelQuintupletIndex][5] = quintuplets.hitIndices()[t5Index][1];
    pixelQuintuplets.hitIndices()[pixelQuintupletIndex][6] = quintuplets.hitIndices()[t5Index][2];
    pixelQuintuplets.hitIndices()[pixelQuintupletIndex][7] = quintuplets.hitIndices()[t5Index][3];
    pixelQuintuplets.hitIndices()[pixelQuintupletIndex][8] = quintuplets.hitIndices()[t5Index][4];
    pixelQuintuplets.hitIndices()[pixelQuintupletIndex][9] = quintuplets.hitIndices()[t5Index][5];
    pixelQuintuplets.hitIndices()[pixelQuintupletIndex][10] = quintuplets.hitIndices()[t5Index][6];
    pixelQuintuplets.hitIndices()[pixelQuintupletIndex][11] = quintuplets.hitIndices()[t5Index][7];
    pixelQuintuplets.hitIndices()[pixelQuintupletIndex][12] = quintuplets.hitIndices()[t5Index][8];
    pixelQuintuplets.hitIndices()[pixelQuintupletIndex][13] = quintuplets.hitIndices()[t5Index][9];

    pixelQuintuplets.rzChiSquared()[pixelQuintupletIndex] = rzChiSquared;
    pixelQuintuplets.rPhiChiSquared()[pixelQuintupletIndex] = rPhiChiSquared;
    pixelQuintuplets.rPhiChiSquaredInwards()[pixelQuintupletIndex] = rPhiChiSquaredInwards;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT5RZChiSquaredCuts(ModulesConst modules,
                                                              uint16_t lowerModuleIndex1,
                                                              uint16_t lowerModuleIndex2,
                                                              uint16_t lowerModuleIndex3,
                                                              uint16_t lowerModuleIndex4,
                                                              uint16_t lowerModuleIndex5,
                                                              float rzChiSquared) {
    const int layer1 =
        modules.layers()[lowerModuleIndex1] + 6 * (modules.subdets()[lowerModuleIndex1] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex1] == Endcap and modules.moduleType()[lowerModuleIndex1] == TwoS);
    const int layer2 =
        modules.layers()[lowerModuleIndex2] + 6 * (modules.subdets()[lowerModuleIndex2] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex2] == Endcap and modules.moduleType()[lowerModuleIndex2] == TwoS);
    const int layer3 =
        modules.layers()[lowerModuleIndex3] + 6 * (modules.subdets()[lowerModuleIndex3] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex3] == Endcap and modules.moduleType()[lowerModuleIndex3] == TwoS);
    const int layer4 =
        modules.layers()[lowerModuleIndex4] + 6 * (modules.subdets()[lowerModuleIndex4] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex4] == Endcap and modules.moduleType()[lowerModuleIndex4] == TwoS);
    const int layer5 =
        modules.layers()[lowerModuleIndex5] + 6 * (modules.subdets()[lowerModuleIndex5] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex5] == Endcap and modules.moduleType()[lowerModuleIndex5] == TwoS);

    if (layer1 == 1 and layer2 == 2 and layer3 == 3) {
      if (layer4 == 12 and layer5 == 13) {
        return rzChiSquared < 451.141f;
      } else if (layer4 == 4 and layer5 == 12) {
        return rzChiSquared < 392.654f;
      } else if (layer4 == 4 and layer5 == 5) {
        return rzChiSquared < 225.322f;
      } else if (layer4 == 7 and layer5 == 13) {
        return rzChiSquared < 595.546f;
      } else if (layer4 == 7 and layer5 == 8) {
        return rzChiSquared < 196.111f;
      }
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 7) {
      if (layer4 == 13 and layer5 == 14) {
        return rzChiSquared < 297.446f;
      } else if (layer4 == 8 and layer5 == 14) {
        return rzChiSquared < 451.141f;
      } else if (layer4 == 8 and layer5 == 9) {
        return rzChiSquared < 518.339f;
      }
    } else if (layer1 == 1 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 9 and layer5 == 10) {
        return rzChiSquared < 341.75f;
      } else if (layer4 == 9 and layer5 == 15) {
        return rzChiSquared < 341.75f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 4) {
      if (layer4 == 12 and layer5 == 13) {
        return rzChiSquared < 392.655f;
      } else if (layer4 == 5 and layer5 == 12) {
        return rzChiSquared < 341.75f;
      } else if (layer4 == 5 and layer5 == 6) {
        return rzChiSquared < 112.537f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer4 == 7) {
      if (layer4 == 13 and layer5 == 14) {
        return rzChiSquared < 595.545f;
      } else if (layer4 == 8 and layer5 == 14) {
        return rzChiSquared < 74.198f;
      }
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 14 and layer5 == 15) {
        return rzChiSquared < 518.339f;
      } else if (layer4 == 9 and layer5 == 10) {
        return rzChiSquared < 8.046f;
      } else if (layer4 == 9 and layer5 == 15) {
        return rzChiSquared < 451.141f;
      }
    } else if (layer1 == 3 and layer2 == 7 and layer3 == 8 and layer4 == 14 and layer5 == 15) {
      return rzChiSquared < 56.207f;
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 9) {
      if (layer4 == 10 and layer5 == 11) {
        return rzChiSquared < 64.578f;
      } else if (layer4 == 10 and layer5 == 16) {
        return rzChiSquared < 85.250f;
      } else if (layer4 == 15 and layer5 == 16) {
        return rzChiSquared < 85.250f;
      }
    }
    return true;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT5RPhiChiSquaredCuts(ModulesConst modules,
                                                                uint16_t lowerModuleIndex1,
                                                                uint16_t lowerModuleIndex2,
                                                                uint16_t lowerModuleIndex3,
                                                                uint16_t lowerModuleIndex4,
                                                                uint16_t lowerModuleIndex5,
                                                                float rPhiChiSquared) {
    const int layer1 =
        modules.layers()[lowerModuleIndex1] + 6 * (modules.subdets()[lowerModuleIndex1] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex1] == Endcap and modules.moduleType()[lowerModuleIndex1] == TwoS);
    const int layer2 =
        modules.layers()[lowerModuleIndex2] + 6 * (modules.subdets()[lowerModuleIndex2] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex2] == Endcap and modules.moduleType()[lowerModuleIndex2] == TwoS);
    const int layer3 =
        modules.layers()[lowerModuleIndex3] + 6 * (modules.subdets()[lowerModuleIndex3] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex3] == Endcap and modules.moduleType()[lowerModuleIndex3] == TwoS);
    const int layer4 =
        modules.layers()[lowerModuleIndex4] + 6 * (modules.subdets()[lowerModuleIndex4] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex4] == Endcap and modules.moduleType()[lowerModuleIndex4] == TwoS);
    const int layer5 =
        modules.layers()[lowerModuleIndex5] + 6 * (modules.subdets()[lowerModuleIndex5] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex5] == Endcap and modules.moduleType()[lowerModuleIndex5] == TwoS);

    if (layer1 == 1 and layer2 == 2 and layer3 == 3) {
      if (layer4 == 12 and layer5 == 13) {
        return rPhiChiSquared < 48.921f;
      } else if (layer4 == 4 and layer5 == 12) {
        return rPhiChiSquared < 97.948f;
      } else if (layer4 == 4 and layer5 == 5) {
        return rPhiChiSquared < 129.3f;
      } else if (layer4 == 7 and layer5 == 13) {
        return rPhiChiSquared < 56.21f;
      } else if (layer4 == 7 and layer5 == 8) {
        return rPhiChiSquared < 74.198f;
      }
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 7) {
      if (layer4 == 13 and layer5 == 14) {
        return rPhiChiSquared < 21.265f;
      } else if (layer4 == 8 and layer5 == 14) {
        return rPhiChiSquared < 37.058f;
      } else if (layer4 == 8 and layer5 == 9) {
        return rPhiChiSquared < 42.578f;
      }
    } else if (layer1 == 1 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 9 and layer5 == 10) {
        return rPhiChiSquared < 32.253f;
      } else if (layer4 == 9 and layer5 == 15) {
        return rPhiChiSquared < 37.058f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 4) {
      if (layer4 == 12 and layer5 == 13) {
        return rPhiChiSquared < 97.947f;
      } else if (layer4 == 5 and layer5 == 12) {
        return rPhiChiSquared < 129.3f;
      } else if (layer4 == 5 and layer5 == 6) {
        return rPhiChiSquared < 170.68f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 7) {
      if (layer4 == 13 and layer5 == 14) {
        return rPhiChiSquared < 48.92f;
      } else if (layer4 == 8 and layer5 == 14) {
        return rPhiChiSquared < 74.2f;
      }
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 14 and layer5 == 15) {
        return rPhiChiSquared < 42.58f;
      } else if (layer4 == 9 and layer5 == 10) {
        return rPhiChiSquared < 37.06f;
      } else if (layer4 == 9 and layer5 == 15) {
        return rPhiChiSquared < 48.92f;
      }
    } else if (layer1 == 3 and layer2 == 7 and layer3 == 8 and layer4 == 14 and layer5 == 15) {
      return rPhiChiSquared < 85.25f;
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 9) {
      if (layer4 == 10 and layer5 == 11) {
        return rPhiChiSquared < 42.58f;
      } else if (layer4 == 10 and layer5 == 16) {
        return rPhiChiSquared < 37.06f;
      } else if (layer4 == 15 and layer5 == 16) {
        return rPhiChiSquared < 37.06f;
      }
    }
    return true;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computeChiSquaredpT5(TAcc const& acc,
                                                            unsigned int nPoints,
                                                            float* xs,
                                                            float* ys,
                                                            float* delta1,
                                                            float* delta2,
                                                            float* slopes,
                                                            bool* isFlat,
                                                            float g,
                                                            float f,
                                                            float radius) {
    /*
    Given values of (g, f, radius) and a set of points (and its uncertainties) compute chi squared
    */
    float c = g * g + f * f - radius * radius;
    float chiSquared = 0.f;
    float absArctanSlope, angleM, xPrime, yPrime, sigma2;
    for (size_t i = 0; i < nPoints; i++) {
      absArctanSlope = ((slopes[i] != kVerticalModuleSlope) ? alpaka::math::abs(acc, alpaka::math::atan(acc, slopes[i]))
                                                            : kPi / 2.f);
      if (xs[i] > 0 and ys[i] > 0) {
        angleM = kPi / 2.f - absArctanSlope;
      } else if (xs[i] < 0 and ys[i] > 0) {
        angleM = absArctanSlope + kPi / 2.f;
      } else if (xs[i] < 0 and ys[i] < 0) {
        angleM = -(absArctanSlope + kPi / 2.f);
      } else if (xs[i] > 0 and ys[i] < 0) {
        angleM = -(kPi / 2.f - absArctanSlope);
      } else {
        angleM = 0;
      }
      if (not isFlat[i]) {
        xPrime = xs[i] * alpaka::math::cos(acc, angleM) + ys[i] * alpaka::math::sin(acc, angleM);
        yPrime = ys[i] * alpaka::math::cos(acc, angleM) - xs[i] * alpaka::math::sin(acc, angleM);
      } else {
        xPrime = xs[i];
        yPrime = ys[i];
      }
      sigma2 = 4 * ((xPrime * delta1[i]) * (xPrime * delta1[i]) + (yPrime * delta2[i]) * (yPrime * delta2[i]));
      chiSquared += (xs[i] * xs[i] + ys[i] * ys[i] - 2 * g * xs[i] - 2 * f * ys[i] + c) *
                    (xs[i] * xs[i] + ys[i] * ys[i] - 2 * g * xs[i] - 2 * f * ys[i] + c) / (sigma2);
    }
    return chiSquared;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void computeSigmasForRegression_pT5(TAcc const& acc,
                                                                     ModulesConst modules,
                                                                     const uint16_t* lowerModuleIndices,
                                                                     float* delta1,
                                                                     float* delta2,
                                                                     float* slopes,
                                                                     bool* isFlat,
                                                                     unsigned int nPoints = 5,
                                                                     bool anchorHits = true) {
    /*
    bool anchorHits required to deal with a weird edge case wherein
    the hits ultimately used in the regression are anchor hits, but the
    lower modules need not all be Pixel Modules (in case of PS). Similarly,
    when we compute the chi squared for the non-anchor hits, the "partner module"
    need not always be a PS strip module, but all non-anchor hits sit on strip
    modules.
    */
    ModuleType moduleType;
    short moduleSubdet, moduleSide;
    float inv1 = kWidthPS / kWidth2S;
    float inv2 = kPixelPSZpitch / kWidth2S;
    float inv3 = kStripPSZpitch / kWidth2S;
    for (size_t i = 0; i < nPoints; i++) {
      moduleType = modules.moduleType()[lowerModuleIndices[i]];
      moduleSubdet = modules.subdets()[lowerModuleIndices[i]];
      moduleSide = modules.sides()[lowerModuleIndices[i]];
      const float& drdz = modules.drdzs()[lowerModuleIndices[i]];
      slopes[i] = modules.dxdys()[lowerModuleIndices[i]];
      //category 1 - barrel PS flat
      if (moduleSubdet == Barrel and moduleType == PS and moduleSide == Center) {
        delta1[i] = inv1;
        delta2[i] = inv1;
        slopes[i] = -999.f;
        isFlat[i] = true;
      }
      //category 2 - barrel 2S
      else if (moduleSubdet == Barrel and moduleType == TwoS) {
        delta1[i] = 1.f;
        delta2[i] = 1.f;
        slopes[i] = -999.f;
        isFlat[i] = true;
      }
      //category 3 - barrel PS tilted
      else if (moduleSubdet == Barrel and moduleType == PS and moduleSide != Center) {
        delta1[i] = inv1;
        isFlat[i] = false;

        if (anchorHits) {
          delta2[i] = (inv2 * drdz / alpaka::math::sqrt(acc, 1 + drdz * drdz));
        } else {
          delta2[i] = (inv3 * drdz / alpaka::math::sqrt(acc, 1 + drdz * drdz));
        }
      }
      //category 4 - endcap PS
      else if (moduleSubdet == Endcap and moduleType == PS) {
        delta1[i] = inv1;
        isFlat[i] = false;
        /*
        despite the type of the module layer of the lower module index,
        all anchor hits are on the pixel side and all non-anchor hits are
        on the strip side!
        */
        if (anchorHits) {
          delta2[i] = inv2;
        } else {
          delta2[i] = inv3;
        }
      }
      //category 5 - endcap 2S
      else if (moduleSubdet == Endcap and moduleType == TwoS) {
        delta1[i] = 1.f;
        delta2[i] = 500.f * inv1;
        isFlat[i] = false;
      }
#ifdef WARNINGS
      else {
        printf("ERROR!!!!! I SHOULDN'T BE HERE!!!! subdet = %d, type = %d, side = %d\n",
               moduleSubdet,
               moduleType,
               moduleSide);
      }
#endif
    }
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computePT5RPhiChiSquared(TAcc const& acc,
                                                                ModulesConst modules,
                                                                uint16_t* lowerModuleIndices,
                                                                float g,
                                                                float f,
                                                                float radius,
                                                                float* xs,
                                                                float* ys) {
    /*
    Compute circle parameters from 3 pixel hits, and then use them to compute the chi squared for the outer hits
    */

    float delta1[5], delta2[5], slopes[5];
    bool isFlat[5];
    float chiSquared = 0;

    computeSigmasForRegression_pT5(acc, modules, lowerModuleIndices, delta1, delta2, slopes, isFlat);
    chiSquared = computeChiSquaredpT5(acc, 5, xs, ys, delta1, delta2, slopes, isFlat, g, f, radius);

    return chiSquared;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computePT5RPhiChiSquaredInwards(
      float g, float f, float r, float* xPix, float* yPix) {
    /*
    Using the computed regression center and radius, compute the chi squared for the pixels
    */

    float chiSquared = 0;
    for (size_t i = 0; i < 2; i++) {
      float residual = (xPix[i] - g) * (xPix[i] - g) + (yPix[i] - f) * (yPix[i] - f) - r * r;
      chiSquared += residual * residual;
    }
    chiSquared *= 0.5f;
    return chiSquared;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT5RPhiChiSquaredInwardsCuts(ModulesConst modules,
                                                                       uint16_t lowerModuleIndex1,
                                                                       uint16_t lowerModuleIndex2,
                                                                       uint16_t lowerModuleIndex3,
                                                                       uint16_t lowerModuleIndex4,
                                                                       uint16_t lowerModuleIndex5,
                                                                       float rPhiChiSquared) {
    const int layer1 =
        modules.layers()[lowerModuleIndex1] + 6 * (modules.subdets()[lowerModuleIndex1] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex1] == Endcap and modules.moduleType()[lowerModuleIndex1] == TwoS);
    const int layer2 =
        modules.layers()[lowerModuleIndex2] + 6 * (modules.subdets()[lowerModuleIndex2] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex2] == Endcap and modules.moduleType()[lowerModuleIndex2] == TwoS);
    const int layer3 =
        modules.layers()[lowerModuleIndex3] + 6 * (modules.subdets()[lowerModuleIndex3] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex3] == Endcap and modules.moduleType()[lowerModuleIndex3] == TwoS);
    const int layer4 =
        modules.layers()[lowerModuleIndex4] + 6 * (modules.subdets()[lowerModuleIndex4] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex4] == Endcap and modules.moduleType()[lowerModuleIndex4] == TwoS);
    const int layer5 =
        modules.layers()[lowerModuleIndex5] + 6 * (modules.subdets()[lowerModuleIndex5] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex5] == Endcap and modules.moduleType()[lowerModuleIndex5] == TwoS);

    if (layer1 == 1 and layer2 == 2 and layer3 == 3) {
      if (layer4 == 12 and layer5 == 13) {
        return rPhiChiSquared < 451.141f;
      } else if (layer4 == 4 and layer5 == 12) {
        return rPhiChiSquared < 786.173f;
      } else if (layer4 == 4 and layer5 == 5) {
        return rPhiChiSquared < 595.545f;
      } else if (layer4 == 7 and layer5 == 13) {
        return rPhiChiSquared < 581.339f;
      } else if (layer4 == 7 and layer5 == 8) {
        return rPhiChiSquared < 112.537f;
      }
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 7) {
      if (layer4 == 13 and layer5 == 14) {
        return rPhiChiSquared < 225.322f;
      } else if (layer4 == 8 and layer5 == 14) {
        return rPhiChiSquared < 1192.402f;
      } else if (layer4 == 8 and layer5 == 9) {
        return rPhiChiSquared < 786.173f;
      }
    } else if (layer1 == 1 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 9 and layer5 == 10) {
        return rPhiChiSquared < 1037.817f;
      } else if (layer4 == 9 and layer5 == 15) {
        return rPhiChiSquared < 1808.536f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 4) {
      if (layer4 == 12 and layer5 == 13) {
        return rPhiChiSquared < 684.253f;
      } else if (layer4 == 5 and layer5 == 12) {
        return rPhiChiSquared < 684.253f;
      } else if (layer4 == 5 and layer5 == 6) {
        return rPhiChiSquared < 684.253f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 7) {
      if (layer4 == 13 and layer5 == 14) {
        return rPhiChiSquared < 451.141f;
      } else if (layer4 == 8 and layer5 == 14) {
        return rPhiChiSquared < 518.34f;
      }
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 14 and layer5 == 15) {
        return rPhiChiSquared < 2077.92f;
      } else if (layer4 == 9 and layer5 == 10) {
        return rPhiChiSquared < 74.20f;
      } else if (layer4 == 9 and layer5 == 15) {
        return rPhiChiSquared < 1808.536f;
      }
    } else if (layer1 == 3 and layer2 == 7 and layer3 == 8 and layer4 == 14 and layer5 == 15) {
      return rPhiChiSquared < 786.173f;
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 9) {
      if (layer4 == 10 and layer5 == 11) {
        return rPhiChiSquared < 1574.076f;
      } else if (layer4 == 10 and layer5 == 16) {
        return rPhiChiSquared < 5492.11f;
      } else if (layer4 == 15 and layer5 == 16) {
        return rPhiChiSquared < 2743.037f;
      }
    }
    return true;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computePT5RZChiSquared(TAcc const& acc,
                                                              ModulesConst modules,
                                                              uint16_t* lowerModuleIndices,
                                                              float* rtPix,
                                                              float* zPix,
                                                              float* rts,
                                                              float* zs) {
    //use the two anchor hits of the pixel segment to compute the slope
    //then compute the pseudo chi squared of the five outer hits

    float slope = (zPix[1] - zPix[0]) / (rtPix[1] - rtPix[0]);
    float residual = 0;
    float error2 = 0;
    //hardcoded array indices!!!
    float RMSE = 0;
    for (size_t i = 0; i < Params_T5::kLayers; i++) {
      uint16_t& lowerModuleIndex = lowerModuleIndices[i];
      const int moduleType = modules.moduleType()[lowerModuleIndex];
      const int moduleSide = modules.sides()[lowerModuleIndex];
      const int moduleSubdet = modules.subdets()[lowerModuleIndex];

      residual = (moduleSubdet == Barrel) ? (zs[i] - zPix[0]) - slope * (rts[i] - rtPix[0])
                                          : (rts[i] - rtPix[0]) - (zs[i] - zPix[0]) / slope;
      const float& drdz = modules.drdzs()[lowerModuleIndex];
      //PS Modules
      if (moduleType == 0) {
        error2 = kPixelPSZpitch * kPixelPSZpitch;
      } else  //2S modules
      {
        error2 = kStrip2SZpitch * kStrip2SZpitch;
      }

      //special dispensation to tilted PS modules!
      if (moduleType == 0 and moduleSubdet == Barrel and moduleSide != Center) {
        error2 /= (1.f + drdz * drdz);
      }
      RMSE += (residual * residual) / error2;
    }

    RMSE = alpaka::math::sqrt(acc, 0.2f * RMSE);  // Divided by the degree of freedom 5.
    return RMSE;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runPixelQuintupletDefaultAlgo(TAcc const& acc,
                                                                    ModulesConst modules,
                                                                    ObjectRangesConst ranges,
                                                                    MiniDoubletsConst mds,
                                                                    SegmentsConst segments,
                                                                    SegmentsPixelConst segmentsPixel,
                                                                    TripletsConst triplets,
                                                                    QuintupletsConst quintuplets,
                                                                    unsigned int pixelSegmentIndex,
                                                                    unsigned int quintupletIndex,
                                                                    float& rzChiSquared,
                                                                    float& rPhiChiSquared,
                                                                    float& rPhiChiSquaredInwards,
                                                                    float& pixelRadius,
                                                                    float& quintupletRadius,
                                                                    float& centerX,
                                                                    float& centerY,
                                                                    unsigned int pixelSegmentArrayIndex) {
    unsigned int t5InnerT3Index = quintuplets.tripletIndices()[quintupletIndex][0];
    unsigned int t5OuterT3Index = quintuplets.tripletIndices()[quintupletIndex][1];

    float pixelRadiusTemp, tripletRadius, rPhiChiSquaredTemp, rzChiSquaredTemp, rPhiChiSquaredInwardsTemp, centerXTemp,
        centerYTemp;

    if (not runPixelTripletDefaultAlgo(acc,
                                       modules,
                                       ranges,
                                       mds,
                                       segments,
                                       segmentsPixel,
                                       triplets,
                                       pixelSegmentIndex,
                                       t5InnerT3Index,
                                       pixelRadiusTemp,
                                       tripletRadius,
                                       centerXTemp,
                                       centerYTemp,
                                       rzChiSquaredTemp,
                                       rPhiChiSquaredTemp,
                                       rPhiChiSquaredInwardsTemp,
                                       false))
      return false;

    unsigned int firstSegmentIndex = triplets.segmentIndices()[t5InnerT3Index][0];
    unsigned int secondSegmentIndex = triplets.segmentIndices()[t5InnerT3Index][1];
    unsigned int thirdSegmentIndex = triplets.segmentIndices()[t5OuterT3Index][0];
    unsigned int fourthSegmentIndex = triplets.segmentIndices()[t5OuterT3Index][1];

    unsigned int pixelInnerMDIndex = segments.mdIndices()[pixelSegmentIndex][0];
    unsigned int pixelOuterMDIndex = segments.mdIndices()[pixelSegmentIndex][1];
    unsigned int firstMDIndex = segments.mdIndices()[firstSegmentIndex][0];
    unsigned int secondMDIndex = segments.mdIndices()[secondSegmentIndex][0];
    unsigned int thirdMDIndex = segments.mdIndices()[secondSegmentIndex][1];
    unsigned int fourthMDIndex = segments.mdIndices()[thirdSegmentIndex][1];
    unsigned int fifthMDIndex = segments.mdIndices()[fourthSegmentIndex][1];

    uint16_t lowerModuleIndex1 = quintuplets.lowerModuleIndices()[quintupletIndex][0];
    uint16_t lowerModuleIndex2 = quintuplets.lowerModuleIndices()[quintupletIndex][1];
    uint16_t lowerModuleIndex3 = quintuplets.lowerModuleIndices()[quintupletIndex][2];
    uint16_t lowerModuleIndex4 = quintuplets.lowerModuleIndices()[quintupletIndex][3];
    uint16_t lowerModuleIndex5 = quintuplets.lowerModuleIndices()[quintupletIndex][4];

    uint16_t lowerModuleIndices[Params_T5::kLayers] = {
        lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5};

    float zPix[Params_pLS::kLayers] = {mds.anchorZ()[pixelInnerMDIndex], mds.anchorZ()[pixelOuterMDIndex]};
    float rtPix[Params_pLS::kLayers] = {mds.anchorRt()[pixelInnerMDIndex], mds.anchorRt()[pixelOuterMDIndex]};
    float zs[Params_T5::kLayers] = {mds.anchorZ()[firstMDIndex],
                                    mds.anchorZ()[secondMDIndex],
                                    mds.anchorZ()[thirdMDIndex],
                                    mds.anchorZ()[fourthMDIndex],
                                    mds.anchorZ()[fifthMDIndex]};
    float rts[Params_T5::kLayers] = {mds.anchorRt()[firstMDIndex],
                                     mds.anchorRt()[secondMDIndex],
                                     mds.anchorRt()[thirdMDIndex],
                                     mds.anchorRt()[fourthMDIndex],
                                     mds.anchorRt()[fifthMDIndex]};

    rzChiSquared = computePT5RZChiSquared(acc, modules, lowerModuleIndices, rtPix, zPix, rts, zs);

    if (/*pixelRadius*/ 0 < 5.0f * kR1GeVf) {  // FIXME: pixelRadius is not defined yet
      if (not passPT5RZChiSquaredCuts(modules,
                                      lowerModuleIndex1,
                                      lowerModuleIndex2,
                                      lowerModuleIndex3,
                                      lowerModuleIndex4,
                                      lowerModuleIndex5,
                                      rzChiSquared))
        return false;
    }

    //outer T5
    float xs[Params_T5::kLayers] = {mds.anchorX()[firstMDIndex],
                                    mds.anchorX()[secondMDIndex],
                                    mds.anchorX()[thirdMDIndex],
                                    mds.anchorX()[fourthMDIndex],
                                    mds.anchorX()[fifthMDIndex]};
    float ys[Params_T5::kLayers] = {mds.anchorY()[firstMDIndex],
                                    mds.anchorY()[secondMDIndex],
                                    mds.anchorY()[thirdMDIndex],
                                    mds.anchorY()[fourthMDIndex],
                                    mds.anchorY()[fifthMDIndex]};

    //get the appropriate radii and centers
    centerX = segmentsPixel.circleCenterX()[pixelSegmentArrayIndex];
    centerY = segmentsPixel.circleCenterY()[pixelSegmentArrayIndex];
    pixelRadius = segmentsPixel.circleRadius()[pixelSegmentArrayIndex];

    float T5CenterX = quintuplets.regressionG()[quintupletIndex];
    float T5CenterY = quintuplets.regressionF()[quintupletIndex];
    quintupletRadius = quintuplets.regressionRadius()[quintupletIndex];

    rPhiChiSquared = computePT5RPhiChiSquared(acc, modules, lowerModuleIndices, centerX, centerY, pixelRadius, xs, ys);

    if (pixelRadius < 5.0f * kR1GeVf) {
      if (not passPT5RPhiChiSquaredCuts(modules,
                                        lowerModuleIndex1,
                                        lowerModuleIndex2,
                                        lowerModuleIndex3,
                                        lowerModuleIndex4,
                                        lowerModuleIndex5,
                                        rPhiChiSquared))
        return false;
    }

    float xPix[] = {mds.anchorX()[pixelInnerMDIndex], mds.anchorX()[pixelOuterMDIndex]};
    float yPix[] = {mds.anchorY()[pixelInnerMDIndex], mds.anchorY()[pixelOuterMDIndex]};
    rPhiChiSquaredInwards = computePT5RPhiChiSquaredInwards(T5CenterX, T5CenterY, quintupletRadius, xPix, yPix);

    if (quintuplets.regressionRadius()[quintupletIndex] < 5.0f * kR1GeVf) {
      if (not passPT5RPhiChiSquaredInwardsCuts(modules,
                                               lowerModuleIndex1,
                                               lowerModuleIndex2,
                                               lowerModuleIndex3,
                                               lowerModuleIndex4,
                                               lowerModuleIndex5,
                                               rPhiChiSquaredInwards))
        return false;
    }
    //trusting the T5 regression center to also be a good estimate..
    centerX = (centerX + T5CenterX) / 2;
    centerY = (centerY + T5CenterY) / 2;

    return true;
  }

  struct CreatePixelQuintupletsFromMap {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  ModulesConst modules,
                                  ModulesPixelConst modulesPixel,
                                  MiniDoubletsConst mds,
                                  SegmentsConst segments,
                                  SegmentsPixel segmentsPixel,
                                  Triplets triplets,
                                  Quintuplets quintuplets,
                                  QuintupletsOccupancyConst quintupletsOccupancy,
                                  PixelQuintuplets pixelQuintuplets,
                                  unsigned int* connectedPixelSize,
                                  unsigned int* connectedPixelIndex,
                                  unsigned int nPixelSegments,
                                  ObjectRangesConst ranges) const {
      auto const globalBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridBlockExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (unsigned int i_pLS = globalThreadIdx[1]; i_pLS < nPixelSegments; i_pLS += gridThreadExtent[1]) {
        auto iLSModule_max = connectedPixelIndex[i_pLS] + connectedPixelSize[i_pLS];
        for (unsigned int iLSModule = connectedPixelIndex[i_pLS] + globalBlockIdx[0]; iLSModule < iLSModule_max;
             iLSModule += gridBlockExtent[0]) {
          //these are actual module indices
          uint16_t quintupletLowerModuleIndex = modulesPixel.connectedPixels()[iLSModule];
          if (quintupletLowerModuleIndex >= modules.nLowerModules())
            continue;
          if (modules.moduleType()[quintupletLowerModuleIndex] == TwoS)
            continue;
          uint16_t pixelModuleIndex = modules.nLowerModules();
          if (segmentsPixel.isDup()[i_pLS])
            continue;
          unsigned int nOuterQuintuplets = quintupletsOccupancy.nQuintuplets()[quintupletLowerModuleIndex];

          if (nOuterQuintuplets == 0)
            continue;

          unsigned int pixelSegmentIndex = ranges.segmentModuleIndices()[pixelModuleIndex] + i_pLS;

          //fetch the quintuplet
          for (unsigned int outerQuintupletArrayIndex = globalThreadIdx[2];
               outerQuintupletArrayIndex < nOuterQuintuplets;
               outerQuintupletArrayIndex += gridThreadExtent[2]) {
            unsigned int quintupletIndex =
                ranges.quintupletModuleIndices()[quintupletLowerModuleIndex] + outerQuintupletArrayIndex;

            if (quintuplets.isDup()[quintupletIndex])
              continue;

            float rzChiSquared, rPhiChiSquared, rPhiChiSquaredInwards, pixelRadius, quintupletRadius, centerX, centerY;

            bool success = runPixelQuintupletDefaultAlgo(acc,
                                                         modules,
                                                         ranges,
                                                         mds,
                                                         segments,
                                                         segmentsPixel,
                                                         triplets,
                                                         quintuplets,
                                                         pixelSegmentIndex,
                                                         quintupletIndex,
                                                         rzChiSquared,
                                                         rPhiChiSquared,
                                                         rPhiChiSquaredInwards,
                                                         pixelRadius,
                                                         quintupletRadius,
                                                         centerX,
                                                         centerY,
                                                         static_cast<unsigned int>(i_pLS));
            if (success) {
              unsigned int totOccupancyPixelQuintuplets = alpaka::atomicAdd(
                  acc, &pixelQuintuplets.totOccupancyPixelQuintuplets(), 1u, alpaka::hierarchy::Threads{});
              if (totOccupancyPixelQuintuplets >= n_max_pixel_quintuplets) {
#ifdef WARNINGS
                printf("Pixel Quintuplet excess alert!\n");
#endif
              } else {
                unsigned int pixelQuintupletIndex =
                    alpaka::atomicAdd(acc, &pixelQuintuplets.nPixelQuintuplets(), 1u, alpaka::hierarchy::Threads{});
                float eta = __H2F(quintuplets.eta()[quintupletIndex]);
                float phi = __H2F(quintuplets.phi()[quintupletIndex]);

                addPixelQuintupletToMemory(modules,
                                           mds,
                                           segments,
                                           quintuplets,
                                           pixelQuintuplets,
                                           pixelSegmentIndex,
                                           quintupletIndex,
                                           pixelQuintupletIndex,
                                           rzChiSquared,
                                           rPhiChiSquared,
                                           rPhiChiSquaredInwards,
                                           rPhiChiSquared,
                                           eta,
                                           phi,
                                           pixelRadius,
                                           quintupletRadius,
                                           centerX,
                                           centerY);

                triplets.partOfPT5()[quintuplets.tripletIndices()[quintupletIndex][0]] = true;
                triplets.partOfPT5()[quintuplets.tripletIndices()[quintupletIndex][1]] = true;
                segmentsPixel.partOfPT5()[i_pLS] = true;
                quintuplets.partOfPT5()[quintupletIndex] = true;
              }  // tot occupancy
            }  // end success
          }  // end T5
        }  // end iLS
      }  // end i_pLS
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
