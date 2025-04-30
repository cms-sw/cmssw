#ifndef RecoTracker_LSTCore_src_alpaka_PixelQuintuplet_h
#define RecoTracker_LSTCore_src_alpaka_PixelQuintuplet_h

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/HitsSoA.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsSoA.h"
#include "RecoTracker/LSTCore/interface/PixelTripletsSoA.h"
#include "RecoTracker/LSTCore/interface/PixelSegmentsSoA.h"
#include "RecoTracker/LSTCore/interface/QuintupletsSoA.h"
#include "RecoTracker/LSTCore/interface/SegmentsSoA.h"
#include "RecoTracker/LSTCore/interface/TripletsSoA.h"

#include "Quintuplet.h"
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

    // These slides show the cut threshold definition. The comments below in the code, e.g, "cat 10", is consistent with the region separation in the slides
    // https://indico.cern.ch/event/1410985/contributions/5931017/attachments/2875400/5035406/helix%20approxi%20for%20pT5%20rzchi2%20new%20results%20versions.pdf
    if (layer1 == 1 and layer2 == 2 and layer3 == 3) {
      if (layer4 == 12 and layer5 == 13) {  // cat 10
        return rzChiSquared < 14.031f;
      } else if (layer4 == 4 and layer5 == 12) {  // cat 12
        return rzChiSquared < 8.760f;
      } else if (layer4 == 4 and layer5 == 5) {  // cat 11
        return rzChiSquared < 3.607f;
      } else if (layer4 == 7 and layer5 == 13) {  // cat 9
        return rzChiSquared < 16.620;
      } else if (layer4 == 7 and layer5 == 8) {  // cat 8
        return rzChiSquared < 17.910f;
      }
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 7) {
      if (layer4 == 13 and layer5 == 14) {  // cat 7
        return rzChiSquared < 8.950f;
      } else if (layer4 == 8 and layer5 == 14) {  // cat 6
        return rzChiSquared < 14.837f;
      } else if (layer4 == 8 and layer5 == 9) {  // cat 5
        return rzChiSquared < 18.519f;
      }
    } else if (layer1 == 1 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 9 and layer5 == 10) {  // cat 3
        return rzChiSquared < 15.093f;
      } else if (layer4 == 9 and layer5 == 15) {  // cat 4
        return rzChiSquared < 11.200f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 4) {
      if (layer4 == 12 and layer5 == 13) {  // cat 20
        return rzChiSquared < 12.868f;
      } else if (layer4 == 5 and layer5 == 12) {  // cat 19
        return rzChiSquared < 6.128f;
      } else if (layer4 == 5 and layer5 == 6) {  // cat 18
        return rzChiSquared < 2.987f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer4 == 7) {
      if (layer4 == 13 and layer5 == 14) {  // cat 17
        return rzChiSquared < 19.446f;
      } else if (layer4 == 8 and layer5 == 14) {  // cat 16
        return rzChiSquared < 17.520f;
      }
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 14 and layer5 == 15) {  // cat 15
        return rzChiSquared < 14.71f;
      } else if (layer4 == 9 and layer5 == 15) {  // cat 14
        return rzChiSquared < 18.213f;
      }
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 9) {
      if (layer4 == 10 and layer5 == 11) {  // cat 0
        return rzChiSquared < 10.016f;
      } else if (layer4 == 10 and layer5 == 16) {  // cat 1
        return rzChiSquared < 87.671f;
      } else if (layer4 == 15 and layer5 == 16) {  // cat 2
        return rzChiSquared < 5.844f;
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

    computeSigmasForRegression(acc, modules, lowerModuleIndices, delta1, delta2, slopes, isFlat);
    chiSquared = computeChiSquared(acc, 5, xs, ys, delta1, delta2, slopes, isFlat, g, f, radius);

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
                                                              const uint16_t* lowerModuleIndices,
                                                              const float* rtPix,
                                                              const float* xPix,
                                                              const float* yPix,
                                                              const float* zPix,
                                                              const float* rts,
                                                              const float* zs,
                                                              float pixelSegmentPt,
                                                              float pixelSegmentPx,
                                                              float pixelSegmentPy,
                                                              float pixelSegmentPz,
                                                              int pixelSegmentCharge) {
    float residual = 0;
    float error2 = 0;
    float RMSE = 0;

    // the pixel positions are in unit of cm, and need to be divided by 100 to be in consistent with unit mm.
    float Px = pixelSegmentPx, Py = pixelSegmentPy, Pz = pixelSegmentPz;
    int charge = pixelSegmentCharge;
    float x1 = xPix[1] / 100;
    float y1 = yPix[1] / 100;
    float z1 = zPix[1] / 100;
    float r1 = rtPix[1] / 100;

    float a = -100 / kR1GeVf * charge;

    for (size_t i = 0; i < Params_T5::kLayers; i++) {
      float zsi = zs[i] / 100;
      float rtsi = rts[i] / 100;
      uint16_t lowerModuleIndex = lowerModuleIndices[i];
      const int moduleType = modules.moduleType()[lowerModuleIndex];
      const int moduleSide = modules.sides()[lowerModuleIndex];
      const int moduleSubdet = modules.subdets()[lowerModuleIndex];

      // calculation is detailed documented here https://indico.cern.ch/event/1185895/contributions/4982756/attachments/2526561/4345805/helix%20pT3%20summarize.pdf
      float diffr, diffz;
      float p = alpaka::math::sqrt(acc, Px * Px + Py * Py + Pz * Pz);

      float rou = a / p;
      if (moduleSubdet == Endcap) {
        float s = (zsi - z1) * p / Pz;
        float x = x1 + Px / a * alpaka::math::sin(acc, rou * s) - Py / a * (1 - alpaka::math::cos(acc, rou * s));
        float y = y1 + Py / a * alpaka::math::sin(acc, rou * s) + Px / a * (1 - alpaka::math::cos(acc, rou * s));
        diffr = alpaka::math::abs(acc, rtsi - alpaka::math::sqrt(acc, x * x + y * y)) * 100;
        residual = diffr;
      }

      if (moduleSubdet == Barrel) {
        float paraA = r1 * r1 + 2 * (Px * Px + Py * Py) / (a * a) + 2 * (y1 * Px - x1 * Py) / a - rtsi * rtsi;
        float paraB = 2 * (x1 * Px + y1 * Py) / a;
        float paraC = 2 * (y1 * Px - x1 * Py) / a + 2 * (Px * Px + Py * Py) / (a * a);
        float A = paraB * paraB + paraC * paraC;
        float B = 2 * paraA * paraB;
        float C = paraA * paraA - paraC * paraC;
        float sol1 = (-B + alpaka::math::sqrt(acc, B * B - 4 * A * C)) / (2 * A);
        float sol2 = (-B - alpaka::math::sqrt(acc, B * B - 4 * A * C)) / (2 * A);
        float solz1 = alpaka::math::asin(acc, sol1) / rou * Pz / p + z1;
        float solz2 = alpaka::math::asin(acc, sol2) / rou * Pz / p + z1;
        float diffz1 = alpaka::math::abs(acc, solz1 - zsi) * 100;
        float diffz2 = alpaka::math::abs(acc, solz2 - zsi) * 100;
        diffz = alpaka::math::min(acc, diffz1, diffz2);
        residual = diffz;
      }

      //PS Modules
      if (moduleType == 0) {
        error2 = kPixelPSZpitch * kPixelPSZpitch;
      } else  //2S modules
      {
        error2 = kStrip2SZpitch * kStrip2SZpitch;
      }

      //special dispensation to tilted PS modules!
      if (moduleType == 0 and moduleSubdet == Barrel and moduleSide != Center) {
        float drdz = modules.drdzs()[lowerModuleIndex];
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
                                                                    PixelSegmentsConst pixelSegments,
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
                                                                    unsigned int pixelSegmentArrayIndex,
                                                                    const float ptCut) {
    unsigned int t5InnerT3Index = quintuplets.tripletIndices()[quintupletIndex][0];
    unsigned int t5OuterT3Index = quintuplets.tripletIndices()[quintupletIndex][1];

    float pixelRadiusTemp, tripletRadius, rPhiChiSquaredTemp, rzChiSquaredTemp, rPhiChiSquaredInwardsTemp, centerXTemp,
        centerYTemp, pixelRadiusErrorTemp;

    if (not runPixelTripletDefaultAlgo(acc,
                                       modules,
                                       ranges,
                                       mds,
                                       segments,
                                       pixelSegments,
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
                                       pixelRadiusErrorTemp,
                                       ptCut,
                                       true,
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

    float rtPix[Params_pLS::kLayers] = {mds.anchorRt()[pixelInnerMDIndex], mds.anchorRt()[pixelOuterMDIndex]};
    float xPix[Params_pLS::kLayers] = {mds.anchorX()[pixelInnerMDIndex], mds.anchorX()[pixelOuterMDIndex]};
    float yPix[Params_pLS::kLayers] = {mds.anchorY()[pixelInnerMDIndex], mds.anchorY()[pixelOuterMDIndex]};
    float zPix[Params_pLS::kLayers] = {mds.anchorZ()[pixelInnerMDIndex], mds.anchorZ()[pixelOuterMDIndex]};
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

    float pixelSegmentPt = pixelSegments.ptIn()[pixelSegmentArrayIndex];
    float pixelSegmentPx = pixelSegments.px()[pixelSegmentArrayIndex];
    float pixelSegmentPy = pixelSegments.py()[pixelSegmentArrayIndex];
    float pixelSegmentPz = pixelSegments.pz()[pixelSegmentArrayIndex];
    int pixelSegmentCharge = pixelSegments.charge()[pixelSegmentArrayIndex];

    rzChiSquared = 0;

    //get the appropriate centers
    pixelRadius = pixelSegments.circleRadius()[pixelSegmentArrayIndex];

    rzChiSquared = computePT5RZChiSquared(acc,
                                          modules,
                                          lowerModuleIndices,
                                          rtPix,
                                          xPix,
                                          yPix,
                                          zPix,
                                          rts,
                                          zs,
                                          pixelSegmentPt,
                                          pixelSegmentPx,
                                          pixelSegmentPy,
                                          pixelSegmentPz,
                                          pixelSegmentCharge);

    if (pixelRadius < 5.0f * kR1GeVf) {  //only apply r-z chi2 cuts for <5GeV tracks
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

    //get the appropriate centers
    centerX = pixelSegments.circleCenterX()[pixelSegmentArrayIndex];
    centerY = pixelSegments.circleCenterY()[pixelSegmentArrayIndex];

    float T5CenterX = quintuplets.regressionCenterX()[quintupletIndex];
    float T5CenterY = quintuplets.regressionCenterY()[quintupletIndex];
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
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  ModulesConst modules,
                                  ModulesPixelConst modulesPixel,
                                  MiniDoubletsConst mds,
                                  SegmentsConst segments,
                                  PixelSegments pixelSegments,
                                  Triplets triplets,
                                  Quintuplets quintuplets,
                                  QuintupletsOccupancyConst quintupletsOccupancy,
                                  PixelQuintuplets pixelQuintuplets,
                                  unsigned int* connectedPixelSize,
                                  unsigned int* connectedPixelIndex,
                                  unsigned int nPixelSegments,
                                  ObjectRangesConst ranges,
                                  const float ptCut) const {
      for (unsigned int i_pLS : cms::alpakatools::uniform_elements_z(acc, nPixelSegments)) {
        auto iLSModule_max = connectedPixelIndex[i_pLS] + connectedPixelSize[i_pLS];
        for (unsigned int iLSModule :
             cms::alpakatools::uniform_elements_y(acc, connectedPixelIndex[i_pLS], iLSModule_max)) {
          //these are actual module indices
          uint16_t quintupletLowerModuleIndex = modulesPixel.connectedPixels()[iLSModule];
          if (quintupletLowerModuleIndex >= modules.nLowerModules())
            continue;
          if (modules.moduleType()[quintupletLowerModuleIndex] == TwoS)
            continue;
          uint16_t pixelModuleIndex = modules.nLowerModules();
          if (pixelSegments.isDup()[i_pLS])
            continue;
          unsigned int nOuterQuintuplets = quintupletsOccupancy.nQuintuplets()[quintupletLowerModuleIndex];

          if (nOuterQuintuplets == 0)
            continue;

          unsigned int pixelSegmentIndex = ranges.segmentModuleIndices()[pixelModuleIndex] + i_pLS;

          //fetch the quintuplet
          for (unsigned int outerQuintupletArrayIndex : cms::alpakatools::uniform_elements_x(acc, nOuterQuintuplets)) {
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
                                                         pixelSegments,
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
                                                         static_cast<unsigned int>(i_pLS),
                                                         ptCut);
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
                pixelSegments.partOfPT5()[i_pLS] = true;
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
