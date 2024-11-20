#ifndef RecoTracker_LSTCore_src_alpaka_PixelTriplet_h
#define RecoTracker_LSTCore_src_alpaka_PixelTriplet_h

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsSoA.h"
#include "RecoTracker/LSTCore/interface/PixelTripletsSoA.h"
#include "RecoTracker/LSTCore/interface/QuintupletsSoA.h"
#include "RecoTracker/LSTCore/interface/SegmentsSoA.h"
#include "RecoTracker/LSTCore/interface/TripletsSoA.h"

#include "Quintuplet.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runTripletDefaultAlgoPPBB(TAcc const& acc,
                                                                ModulesConst modules,
                                                                ObjectRangesConst ranges,
                                                                MiniDoubletsConst mds,
                                                                SegmentsConst segments,
                                                                SegmentsPixelConst segmentsPixel,
                                                                uint16_t pixelModuleIndex,
                                                                uint16_t outerInnerLowerModuleIndex,
                                                                uint16_t outerOuterLowerModuleIndex,
                                                                unsigned int innerSegmentIndex,
                                                                unsigned int outerSegmentIndex,
                                                                unsigned int firstMDIndex,
                                                                unsigned int secondMDIndex,
                                                                unsigned int thirdMDIndex,
                                                                unsigned int fourthMDIndex);
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runTripletDefaultAlgoPPEE(TAcc const& acc,
                                                                ModulesConst modules,
                                                                ObjectRangesConst ranges,
                                                                MiniDoubletsConst mds,
                                                                SegmentsConst segments,
                                                                SegmentsPixelConst segmentsPixel,
                                                                uint16_t pixelModuleIndex,
                                                                uint16_t outerInnerLowerModuleIndex,
                                                                uint16_t outerOuterLowerModuleIndex,
                                                                unsigned int innerSegmentIndex,
                                                                unsigned int outerSegmentIndex,
                                                                unsigned int firstMDIndex,
                                                                unsigned int secondMDIndex,
                                                                unsigned int thirdMDIndex,
                                                                unsigned int fourthMDIndex);

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addPixelTripletToMemory(MiniDoubletsConst mds,
                                                              SegmentsConst segments,
                                                              TripletsConst triplets,
                                                              PixelTriplets pixelTriplets,
                                                              unsigned int pixelSegmentIndex,
                                                              unsigned int tripletIndex,
                                                              float pixelRadius,
                                                              float tripletRadius,
                                                              float centerX,
                                                              float centerY,
                                                              float rPhiChiSquared,
                                                              float rPhiChiSquaredInwards,
                                                              float rzChiSquared,
                                                              unsigned int pixelTripletIndex,
                                                              float pt,
                                                              float eta,
                                                              float phi,
                                                              float eta_pix,
                                                              float phi_pix,
                                                              float score) {
    pixelTriplets.pixelSegmentIndices()[pixelTripletIndex] = pixelSegmentIndex;
    pixelTriplets.tripletIndices()[pixelTripletIndex] = tripletIndex;
    pixelTriplets.pixelRadius()[pixelTripletIndex] = __F2H(pixelRadius);
    pixelTriplets.tripletRadius()[pixelTripletIndex] = __F2H(tripletRadius);
    pixelTriplets.pt()[pixelTripletIndex] = __F2H(pt);
    pixelTriplets.eta()[pixelTripletIndex] = __F2H(eta);
    pixelTriplets.phi()[pixelTripletIndex] = __F2H(phi);
    pixelTriplets.eta_pix()[pixelTripletIndex] = __F2H(eta_pix);
    pixelTriplets.phi_pix()[pixelTripletIndex] = __F2H(phi_pix);
    pixelTriplets.isDup()[pixelTripletIndex] = false;
    pixelTriplets.score()[pixelTripletIndex] = __F2H(score);

    pixelTriplets.centerX()[pixelTripletIndex] = __F2H(centerX);
    pixelTriplets.centerY()[pixelTripletIndex] = __F2H(centerY);
    pixelTriplets.logicalLayers()[pixelTripletIndex][0] = 0;
    pixelTriplets.logicalLayers()[pixelTripletIndex][1] = 0;
    pixelTriplets.logicalLayers()[pixelTripletIndex][2] = triplets.logicalLayers()[tripletIndex][0];
    pixelTriplets.logicalLayers()[pixelTripletIndex][3] = triplets.logicalLayers()[tripletIndex][1];
    pixelTriplets.logicalLayers()[pixelTripletIndex][4] = triplets.logicalLayers()[tripletIndex][2];

    pixelTriplets.lowerModuleIndices()[pixelTripletIndex][0] = segments.innerLowerModuleIndices()[pixelSegmentIndex];
    pixelTriplets.lowerModuleIndices()[pixelTripletIndex][1] = segments.outerLowerModuleIndices()[pixelSegmentIndex];
    pixelTriplets.lowerModuleIndices()[pixelTripletIndex][2] = triplets.lowerModuleIndices()[tripletIndex][0];
    pixelTriplets.lowerModuleIndices()[pixelTripletIndex][3] = triplets.lowerModuleIndices()[tripletIndex][1];
    pixelTriplets.lowerModuleIndices()[pixelTripletIndex][4] = triplets.lowerModuleIndices()[tripletIndex][2];

    unsigned int pixelInnerMD = segments.mdIndices()[pixelSegmentIndex][0];
    unsigned int pixelOuterMD = segments.mdIndices()[pixelSegmentIndex][1];

    pixelTriplets.hitIndices()[pixelTripletIndex][0] = mds.anchorHitIndices()[pixelInnerMD];
    pixelTriplets.hitIndices()[pixelTripletIndex][1] = mds.outerHitIndices()[pixelInnerMD];
    pixelTriplets.hitIndices()[pixelTripletIndex][2] = mds.anchorHitIndices()[pixelOuterMD];
    pixelTriplets.hitIndices()[pixelTripletIndex][3] = mds.outerHitIndices()[pixelOuterMD];

    pixelTriplets.hitIndices()[pixelTripletIndex][4] = triplets.hitIndices()[tripletIndex][0];
    pixelTriplets.hitIndices()[pixelTripletIndex][5] = triplets.hitIndices()[tripletIndex][1];
    pixelTriplets.hitIndices()[pixelTripletIndex][6] = triplets.hitIndices()[tripletIndex][2];
    pixelTriplets.hitIndices()[pixelTripletIndex][7] = triplets.hitIndices()[tripletIndex][3];
    pixelTriplets.hitIndices()[pixelTripletIndex][8] = triplets.hitIndices()[tripletIndex][4];
    pixelTriplets.hitIndices()[pixelTripletIndex][9] = triplets.hitIndices()[tripletIndex][5];
    pixelTriplets.rPhiChiSquared()[pixelTripletIndex] = rPhiChiSquared;
    pixelTriplets.rPhiChiSquaredInwards()[pixelTripletIndex] = rPhiChiSquaredInwards;
    pixelTriplets.rzChiSquared()[pixelTripletIndex] = rzChiSquared;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runPixelTrackletDefaultAlgopT3(TAcc const& acc,
                                                                     ModulesConst modules,
                                                                     ObjectRangesConst ranges,
                                                                     MiniDoubletsConst mds,
                                                                     SegmentsConst segments,
                                                                     SegmentsPixelConst segmentsPixel,
                                                                     uint16_t pixelLowerModuleIndex,
                                                                     uint16_t outerInnerLowerModuleIndex,
                                                                     uint16_t outerOuterLowerModuleIndex,
                                                                     unsigned int innerSegmentIndex,
                                                                     unsigned int outerSegmentIndex) {
    short outerInnerLowerModuleSubdet = modules.subdets()[outerInnerLowerModuleIndex];
    short outerOuterLowerModuleSubdet = modules.subdets()[outerOuterLowerModuleIndex];

    unsigned int firstMDIndex = segments.mdIndices()[innerSegmentIndex][0];
    unsigned int secondMDIndex = segments.mdIndices()[innerSegmentIndex][1];

    unsigned int thirdMDIndex = segments.mdIndices()[outerSegmentIndex][0];
    unsigned int fourthMDIndex = segments.mdIndices()[outerSegmentIndex][1];

    if (outerInnerLowerModuleSubdet == Barrel and
        (outerOuterLowerModuleSubdet == Barrel or outerOuterLowerModuleSubdet == Endcap)) {
      return runTripletDefaultAlgoPPBB(acc,
                                       modules,
                                       ranges,
                                       mds,
                                       segments,
                                       segmentsPixel,
                                       pixelLowerModuleIndex,
                                       outerInnerLowerModuleIndex,
                                       outerOuterLowerModuleIndex,
                                       innerSegmentIndex,
                                       outerSegmentIndex,
                                       firstMDIndex,
                                       secondMDIndex,
                                       thirdMDIndex,
                                       fourthMDIndex);
    } else if (outerInnerLowerModuleSubdet == Endcap and outerOuterLowerModuleSubdet == Endcap) {
      return runTripletDefaultAlgoPPEE(acc,
                                       modules,
                                       ranges,
                                       mds,
                                       segments,
                                       segmentsPixel,
                                       pixelLowerModuleIndex,
                                       outerInnerLowerModuleIndex,
                                       outerOuterLowerModuleIndex,
                                       innerSegmentIndex,
                                       outerSegmentIndex,
                                       firstMDIndex,
                                       secondMDIndex,
                                       thirdMDIndex,
                                       fourthMDIndex);
    }
    return false;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT3RZChiSquaredCuts(ModulesConst modules,
                                                              uint16_t lowerModuleIndex1,
                                                              uint16_t lowerModuleIndex2,
                                                              uint16_t lowerModuleIndex3,
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

    if (layer1 == 8 and layer2 == 9 and layer3 == 10) {
      return rzChiSquared < 13.6067f;
    } else if (layer1 == 8 and layer2 == 9 and layer3 == 15) {
      return rzChiSquared < 5.5953f;
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 9) {
      return rzChiSquared < 3.9263f;
    }
    /*
    else if(layer1 == 7 and layer2 == 8 and layer3 == 14)
    {   
        // PS+PS+2S in endcap layers 1+2+3, which is not really feasible in the current geometry,
        // without skipping barrel layers 1 and 2 (not allowed by algorithm logic).
    }
    */
    else if (layer1 == 1 and layer2 == 2 and layer3 == 3) {
      return rzChiSquared < 9.4377f;
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 7) {
      return rzChiSquared < 9.9975f;
    } else if (layer1 == 1 and layer2 == 7 and layer3 == 8) {
      return rzChiSquared < 8.6369f;
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 7) {
      return rzChiSquared < 37.945f;
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 12) {
      return rzChiSquared < 43.0167f;
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 4) {
      return rzChiSquared < 8.6923f;
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 8) {
      return rzChiSquared < 11.9672f;
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 13) {
      return rzChiSquared < 16.2133f;
    }

    //default - category not found!
    return true;
  }

  //TODO: merge this one and the pT5 function later into a single function
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computePT3RPhiChiSquared(TAcc const& acc,
                                                                ModulesConst modules,
                                                                uint16_t* lowerModuleIndices,
                                                                float g,
                                                                float f,
                                                                float radius,
                                                                float* xs,
                                                                float* ys) {
    float delta1[3]{}, delta2[3]{}, slopes[3]{};
    bool isFlat[3]{};
    float chiSquared = 0;
    float inv1 = kWidthPS / kWidth2S;
    float inv2 = kPixelPSZpitch / kWidth2S;
    for (size_t i = 0; i < 3; i++) {
      ModuleType moduleType = modules.moduleType()[lowerModuleIndices[i]];
      short moduleSubdet = modules.subdets()[lowerModuleIndices[i]];
      short moduleSide = modules.sides()[lowerModuleIndices[i]];
      float drdz = modules.drdzs()[lowerModuleIndices[i]];
      slopes[i] = modules.dxdys()[lowerModuleIndices[i]];
      //category 1 - barrel PS flat
      if (moduleSubdet == Barrel and moduleType == PS and moduleSide == Center) {
        delta1[i] = inv1;
        delta2[i] = inv1;
        slopes[i] = -999;
        isFlat[i] = true;
      }
      //category 2 - barrel 2S
      else if (moduleSubdet == Barrel and moduleType == TwoS) {
        delta1[i] = 1;
        delta2[i] = 1;
        slopes[i] = -999;
        isFlat[i] = true;
      }
      //category 3 - barrel PS tilted
      else if (moduleSubdet == Barrel and moduleType == PS and moduleSide != Center) {
        delta1[i] = inv1;
        isFlat[i] = false;
        delta2[i] = (inv2 * drdz / alpaka::math::sqrt(acc, 1 + drdz * drdz));
      }
      //category 4 - endcap PS
      else if (moduleSubdet == Endcap and moduleType == PS) {
        delta1[i] = inv1;
        isFlat[i] = false;

        /*
        despite the type of the module layer of the lower module index, all anchor
        hits are on the pixel side and all non-anchor hits are on the strip side!
        */
        delta2[i] = inv2;
      }
      //category 5 - endcap 2S
      else if (moduleSubdet == Endcap and moduleType == TwoS) {
        delta1[i] = 1;
        delta2[i] = 500 * inv1;
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
    chiSquared = computeChiSquared(acc, 3, xs, ys, delta1, delta2, slopes, isFlat, g, f, radius);

    return chiSquared;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computePT3RPhiChiSquaredInwards(
      float g, float f, float r, float* xPix, float* yPix) {
    float residual = (xPix[0] - g) * (xPix[0] - g) + (yPix[0] - f) * (yPix[0] - f) - r * r;
    float chiSquared = residual * residual;
    residual = (xPix[1] - g) * (xPix[1] - g) + (yPix[1] - f) * (yPix[1] - f) - r * r;
    chiSquared += residual * residual;

    chiSquared *= 0.5f;
    return chiSquared;
  }

  //90pc threshold
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT3RPhiChiSquaredCuts(ModulesConst modules,
                                                                uint16_t lowerModuleIndex1,
                                                                uint16_t lowerModuleIndex2,
                                                                uint16_t lowerModuleIndex3,
                                                                float chiSquared) {
    const int layer1 =
        modules.layers()[lowerModuleIndex1] + 6 * (modules.subdets()[lowerModuleIndex1] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex1] == Endcap and modules.moduleType()[lowerModuleIndex1] == TwoS);
    const int layer2 =
        modules.layers()[lowerModuleIndex2] + 6 * (modules.subdets()[lowerModuleIndex2] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex2] == Endcap and modules.moduleType()[lowerModuleIndex2] == TwoS);
    const int layer3 =
        modules.layers()[lowerModuleIndex3] + 6 * (modules.subdets()[lowerModuleIndex3] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex3] == Endcap and modules.moduleType()[lowerModuleIndex3] == TwoS);

    if (layer1 == 8 and layer2 == 9 and layer3 == 10) {
      return chiSquared < 7.003f;
    } else if (layer1 == 8 and layer2 == 9 and layer3 == 15) {
      return chiSquared < 0.5f;
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 9) {
      return chiSquared < 8.046f;
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 14) {
      return chiSquared < 0.575f;
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 7) {
      return chiSquared < 5.304f;
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 3) {
      return chiSquared < 10.6211f;
    } else if (layer1 == 1 and layer2 == 7 and layer3 == 8) {
      return chiSquared < 4.617f;
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 8) {
      return chiSquared < 8.046f;
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 13) {
      return chiSquared < 0.435f;
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 7) {
      return chiSquared < 9.244f;
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 12) {
      return chiSquared < 0.287f;
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 4) {
      return chiSquared < 18.509f;
    }

    return true;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT3RPhiChiSquaredInwardsCuts(ModulesConst modules,
                                                                       uint16_t lowerModuleIndex1,
                                                                       uint16_t lowerModuleIndex2,
                                                                       uint16_t lowerModuleIndex3,
                                                                       float chiSquared) {
    const int layer1 =
        modules.layers()[lowerModuleIndex1] + 6 * (modules.subdets()[lowerModuleIndex1] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex1] == Endcap and modules.moduleType()[lowerModuleIndex1] == TwoS);
    const int layer2 =
        modules.layers()[lowerModuleIndex2] + 6 * (modules.subdets()[lowerModuleIndex2] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex2] == Endcap and modules.moduleType()[lowerModuleIndex2] == TwoS);
    const int layer3 =
        modules.layers()[lowerModuleIndex3] + 6 * (modules.subdets()[lowerModuleIndex3] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex3] == Endcap and modules.moduleType()[lowerModuleIndex3] == TwoS);

    if (layer1 == 7 and layer2 == 8 and layer3 == 9)  // endcap layer 1,2,3, ps
    {
      return chiSquared < 22016.8055f;
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 14)  // endcap layer 1,2,3 layer3->2s
    {
      return chiSquared < 935179.56807f;
    } else if (layer1 == 8 and layer2 == 9 and layer3 == 10)  // endcap layer 2,3,4
    {
      return chiSquared < 29064.12959f;
    } else if (layer1 == 8 and layer2 == 9 and layer3 == 15)  // endcap layer 2,3,4, layer3->2s
    {
      return chiSquared < 935179.5681f;
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 3)  // barrel 1,2,3
    {
      return chiSquared < 1370.0113195101474f;
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 7)  // barrel 1,2 endcap 1
    {
      return chiSquared < 5492.110048314815f;
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 4)  // barrel 2,3,4
    {
      return chiSquared < 4160.410806470067f;
    } else if (layer1 == 1 and layer2 == 7 and layer3 == 8)  // barrel 1, endcap 1,2
    {
      return chiSquared < 29064.129591225726f;
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 7)  // barrel 2,3 endcap 1
    {
      return chiSquared < 12634.215376250893f;
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 12)  // barrel 2,3, endcap 1->2s
    {
      return chiSquared < 353821.69361145404f;
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 8)  // barrel2, endcap 1,2
    {
      return chiSquared < 33393.26076341235f;
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 13)  //barrel 2, endcap 1, endcap2->2s
    {
      return chiSquared < 935179.5680742573f;
    }

    return true;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool checkIntervalOverlappT3(float firstMin,
                                                              float firstMax,
                                                              float secondMin,
                                                              float secondMax) {
    return ((firstMin <= secondMin) && (secondMin < firstMax)) || ((secondMin < firstMin) && (firstMin < secondMax));
  }

  /*bounds for high Pt taken from : http://uaf-10.t2.ucsd.edu/~bsathian/SDL/T5_efficiency/efficiencies/new_efficiencies/efficiencies_20210513_T5_recovering_high_Pt_efficiencies/highE_radius_matching/highE_bounds.txt */
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passRadiusCriterionBBB(TAcc const& acc,
                                                             float pixelRadius,
                                                             float pixelRadiusError,
                                                             float tripletRadius) {
    float tripletInvRadiusErrorBound = 0.15624f;
    float pixelInvRadiusErrorBound = 0.17235f;

    if (pixelRadius > 2.0f * kR1GeVf) {
      pixelInvRadiusErrorBound = 0.6375f;
      tripletInvRadiusErrorBound = 0.6588f;
    }

    float tripletRadiusInvMax = (1 + tripletInvRadiusErrorBound) / tripletRadius;
    float tripletRadiusInvMin = alpaka::math::max(acc, (1 - tripletInvRadiusErrorBound) / tripletRadius, 0.0f);

    float pixelRadiusInvMax =
        alpaka::math::max(acc, (1 + pixelInvRadiusErrorBound) / pixelRadius, 1.f / (pixelRadius - pixelRadiusError));
    float pixelRadiusInvMin =
        alpaka::math::min(acc, (1 - pixelInvRadiusErrorBound) / pixelRadius, 1.f / (pixelRadius + pixelRadiusError));

    return checkIntervalOverlappT3(tripletRadiusInvMin, tripletRadiusInvMax, pixelRadiusInvMin, pixelRadiusInvMax);
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passRadiusCriterionBBE(TAcc const& acc,
                                                             float pixelRadius,
                                                             float pixelRadiusError,
                                                             float tripletRadius) {
    float tripletInvRadiusErrorBound = 0.45972f;
    float pixelInvRadiusErrorBound = 0.19644f;

    if (pixelRadius > 2.0f * kR1GeVf) {
      pixelInvRadiusErrorBound = 0.6805f;
      tripletInvRadiusErrorBound = 0.8557f;
    }

    float tripletRadiusInvMax = (1 + tripletInvRadiusErrorBound) / tripletRadius;
    float tripletRadiusInvMin = alpaka::math::max(acc, (1 - tripletInvRadiusErrorBound) / tripletRadius, 0.0f);

    float pixelRadiusInvMax =
        alpaka::math::max(acc, (1 + pixelInvRadiusErrorBound) / pixelRadius, 1.f / (pixelRadius - pixelRadiusError));
    float pixelRadiusInvMin =
        alpaka::math::min(acc, (1 - pixelInvRadiusErrorBound) / pixelRadius, 1.f / (pixelRadius + pixelRadiusError));

    return checkIntervalOverlappT3(tripletRadiusInvMin, tripletRadiusInvMax, pixelRadiusInvMin, pixelRadiusInvMax);
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passRadiusCriterionBEE(TAcc const& acc,
                                                             float pixelRadius,
                                                             float pixelRadiusError,
                                                             float tripletRadius) {
    float tripletInvRadiusErrorBound = 1.59294f;
    float pixelInvRadiusErrorBound = 0.255181f;

    if (pixelRadius > 2.0f * kR1GeVf)  //as good as not having selections
    {
      pixelInvRadiusErrorBound = 2.2091f;
      tripletInvRadiusErrorBound = 2.3548f;
    }

    float tripletRadiusInvMax = (1 + tripletInvRadiusErrorBound) / tripletRadius;
    float tripletRadiusInvMin = alpaka::math::max(acc, (1 - tripletInvRadiusErrorBound) / tripletRadius, 0.0f);

    float pixelRadiusInvMax =
        alpaka::math::max(acc, (1 + pixelInvRadiusErrorBound) / pixelRadius, 1.f / (pixelRadius - pixelRadiusError));
    float pixelRadiusInvMin =
        alpaka::math::min(acc, (1 - pixelInvRadiusErrorBound) / pixelRadius, 1.f / (pixelRadius + pixelRadiusError));
    pixelRadiusInvMin = alpaka::math::max(acc, pixelRadiusInvMin, 0.0f);

    return checkIntervalOverlappT3(tripletRadiusInvMin, tripletRadiusInvMax, pixelRadiusInvMin, pixelRadiusInvMax);
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passRadiusCriterionEEE(TAcc const& acc,
                                                             float pixelRadius,
                                                             float pixelRadiusError,
                                                             float tripletRadius) {
    float tripletInvRadiusErrorBound = 1.7006f;
    float pixelInvRadiusErrorBound = 0.26367f;

    if (pixelRadius > 2.0f * kR1GeVf)  //as good as not having selections
    {
      pixelInvRadiusErrorBound = 2.286f;
      tripletInvRadiusErrorBound = 2.436f;
    }

    float tripletRadiusInvMax = (1 + tripletInvRadiusErrorBound) / tripletRadius;
    float tripletRadiusInvMin = alpaka::math::max(acc, (1 - tripletInvRadiusErrorBound) / tripletRadius, 0.0f);

    float pixelRadiusInvMax =
        alpaka::math::max(acc, (1 + pixelInvRadiusErrorBound) / pixelRadius, 1.f / (pixelRadius - pixelRadiusError));
    float pixelRadiusInvMin =
        alpaka::math::min(acc, (1 - pixelInvRadiusErrorBound) / pixelRadius, 1.f / (pixelRadius + pixelRadiusError));
    pixelRadiusInvMin = alpaka::math::max(acc, 0.0f, pixelRadiusInvMin);

    return checkIntervalOverlappT3(tripletRadiusInvMin, tripletRadiusInvMax, pixelRadiusInvMin, pixelRadiusInvMax);
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passRadiusCriterion(TAcc const& acc,
                                                          ModulesConst modules,
                                                          float pixelRadius,
                                                          float pixelRadiusError,
                                                          float tripletRadius,
                                                          int16_t lowerModuleIndex,
                                                          uint16_t middleModuleIndex,
                                                          uint16_t upperModuleIndex) {
    if (modules.subdets()[lowerModuleIndex] == Endcap) {
      return passRadiusCriterionEEE(acc, pixelRadius, pixelRadiusError, tripletRadius);
    } else if (modules.subdets()[middleModuleIndex] == Endcap) {
      return passRadiusCriterionBEE(acc, pixelRadius, pixelRadiusError, tripletRadius);
    } else if (modules.subdets()[upperModuleIndex] == Endcap) {
      return passRadiusCriterionBBE(acc, pixelRadius, pixelRadiusError, tripletRadius);
    } else {
      return passRadiusCriterionBBB(acc, pixelRadius, pixelRadiusError, tripletRadius);
    }
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computePT3RZChiSquared(TAcc const& acc,
                                                              ModulesConst modules,
                                                              const uint16_t* lowerModuleIndices,
                                                              const float* rtPix,
                                                              const float* xPix,
                                                              const float* yPix,
                                                              const float* zPix,
                                                              const float* rts,
                                                              const float* xs,
                                                              const float* ys,
                                                              const float* zs,
                                                              float pixelSegmentPt,
                                                              float pixelSegmentPx,
                                                              float pixelSegmentPy,
                                                              float pixelSegmentPz,
                                                              int pixelSegmentCharge) {
    float residual = 0;
    float error2 = 0;
    float RMSE = 0;

    float Px = pixelSegmentPx, Py = pixelSegmentPy, Pz = pixelSegmentPz;
    int charge = pixelSegmentCharge;
    float x1 = xPix[1] / 100;
    float y1 = yPix[1] / 100;
    float z1 = zPix[1] / 100;
    float r1 = rtPix[1] / 100;

    float a = -2.f * k2Rinv1GeVf * 100 * charge;  // multiply by 100 to make the correct length units

    for (size_t i = 0; i < Params_T3::kLayers; i++) {
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
      }

      residual = moduleSubdet == Barrel ? diffz : diffr;

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
        error2 /= (1 + drdz * drdz);
      }
      RMSE += (residual * residual) / error2;
    }

    RMSE = alpaka::math::sqrt(acc, 0.2f * RMSE);  // Divided by the degree of freedom 5.

    return RMSE;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runPixelTripletDefaultAlgo(TAcc const& acc,
                                                                 ModulesConst modules,
                                                                 ObjectRangesConst ranges,
                                                                 MiniDoubletsConst mds,
                                                                 SegmentsConst segments,
                                                                 SegmentsPixelConst segmentsPixel,
                                                                 TripletsConst triplets,
                                                                 unsigned int pixelSegmentIndex,
                                                                 unsigned int tripletIndex,
                                                                 float& pixelRadius,
                                                                 float& tripletRadius,
                                                                 float& centerX,
                                                                 float& centerY,
                                                                 float& rzChiSquared,
                                                                 float& rPhiChiSquared,
                                                                 float& rPhiChiSquaredInwards,
                                                                 bool runChiSquaredCuts = true) {
    //run pT4 compatibility between the pixel segment and inner segment, and between the pixel and outer segment of the triplet
    uint16_t pixelModuleIndex = segments.innerLowerModuleIndices()[pixelSegmentIndex];

    uint16_t lowerModuleIndex = triplets.lowerModuleIndices()[tripletIndex][0];
    uint16_t middleModuleIndex = triplets.lowerModuleIndices()[tripletIndex][1];
    uint16_t upperModuleIndex = triplets.lowerModuleIndices()[tripletIndex][2];

    {
      // pixel segment vs inner segment of the triplet
      if (not runPixelTrackletDefaultAlgopT3(acc,
                                             modules,
                                             ranges,
                                             mds,
                                             segments,
                                             segmentsPixel,
                                             pixelModuleIndex,
                                             lowerModuleIndex,
                                             middleModuleIndex,
                                             pixelSegmentIndex,
                                             triplets.segmentIndices()[tripletIndex][0]))
        return false;

      //pixel segment vs outer segment of triplet
      if (not runPixelTrackletDefaultAlgopT3(acc,
                                             modules,
                                             ranges,
                                             mds,
                                             segments,
                                             segmentsPixel,
                                             pixelModuleIndex,
                                             middleModuleIndex,
                                             upperModuleIndex,
                                             pixelSegmentIndex,
                                             triplets.segmentIndices()[tripletIndex][1]))
        return false;
    }

    //pt matching between the pixel ptin and the triplet circle pt
    unsigned int pixelSegmentArrayIndex = pixelSegmentIndex - ranges.segmentModuleIndices()[pixelModuleIndex];
    float pixelSegmentPt = segmentsPixel.ptIn()[pixelSegmentArrayIndex];
    float pixelSegmentPtError = segmentsPixel.ptErr()[pixelSegmentArrayIndex];
    float pixelSegmentPx = segmentsPixel.px()[pixelSegmentArrayIndex];
    float pixelSegmentPy = segmentsPixel.py()[pixelSegmentArrayIndex];
    float pixelSegmentPz = segmentsPixel.pz()[pixelSegmentArrayIndex];
    int pixelSegmentCharge = segmentsPixel.charge()[pixelSegmentArrayIndex];

    float pixelG = segmentsPixel.circleCenterX()[pixelSegmentArrayIndex];
    float pixelF = segmentsPixel.circleCenterY()[pixelSegmentArrayIndex];
    float pixelRadiusPCA = segmentsPixel.circleRadius()[pixelSegmentArrayIndex];

    unsigned int pixelInnerMDIndex = segments.mdIndices()[pixelSegmentIndex][0];
    unsigned int pixelOuterMDIndex = segments.mdIndices()[pixelSegmentIndex][1];

    pixelRadius = pixelSegmentPt * kR1GeVf;
    float pixelRadiusError = pixelSegmentPtError * kR1GeVf;
    unsigned int tripletInnerSegmentIndex = triplets.segmentIndices()[tripletIndex][0];
    unsigned int tripletOuterSegmentIndex = triplets.segmentIndices()[tripletIndex][1];

    unsigned int firstMDIndex = segments.mdIndices()[tripletInnerSegmentIndex][0];
    unsigned int secondMDIndex = segments.mdIndices()[tripletInnerSegmentIndex][1];
    unsigned int thirdMDIndex = segments.mdIndices()[tripletOuterSegmentIndex][1];

    float xs[Params_T3::kLayers] = {
        mds.anchorX()[firstMDIndex], mds.anchorX()[secondMDIndex], mds.anchorX()[thirdMDIndex]};
    float ys[Params_T3::kLayers] = {
        mds.anchorY()[firstMDIndex], mds.anchorY()[secondMDIndex], mds.anchorY()[thirdMDIndex]};

    float g, f;
    tripletRadius = triplets.radius()[tripletIndex];
    g = triplets.centerX()[tripletIndex];
    f = triplets.centerY()[tripletIndex];

    if (not passRadiusCriterion(acc,
                                modules,
                                pixelRadius,
                                pixelRadiusError,
                                tripletRadius,
                                lowerModuleIndex,
                                middleModuleIndex,
                                upperModuleIndex))
      return false;

    uint16_t lowerModuleIndices[Params_T3::kLayers] = {lowerModuleIndex, middleModuleIndex, upperModuleIndex};

    if (runChiSquaredCuts and pixelSegmentPt < 5.0f) {
      float rts[Params_T3::kLayers] = {
          mds.anchorRt()[firstMDIndex], mds.anchorRt()[secondMDIndex], mds.anchorRt()[thirdMDIndex]};
      float zs[Params_T3::kLayers] = {
          mds.anchorZ()[firstMDIndex], mds.anchorZ()[secondMDIndex], mds.anchorZ()[thirdMDIndex]};
      float rtPix[Params_pLS::kLayers] = {mds.anchorRt()[pixelInnerMDIndex], mds.anchorRt()[pixelOuterMDIndex]};
      float xPix[Params_pLS::kLayers] = {mds.anchorX()[pixelInnerMDIndex], mds.anchorX()[pixelOuterMDIndex]};
      float yPix[Params_pLS::kLayers] = {mds.anchorY()[pixelInnerMDIndex], mds.anchorY()[pixelOuterMDIndex]};
      float zPix[Params_pLS::kLayers] = {mds.anchorZ()[pixelInnerMDIndex], mds.anchorZ()[pixelOuterMDIndex]};

      rzChiSquared = computePT3RZChiSquared(acc,
                                            modules,
                                            lowerModuleIndices,
                                            rtPix,
                                            xPix,
                                            yPix,
                                            zPix,
                                            rts,
                                            xs,
                                            ys,
                                            zs,
                                            pixelSegmentPt,
                                            pixelSegmentPx,
                                            pixelSegmentPy,
                                            pixelSegmentPz,
                                            pixelSegmentCharge);
      if (not passPT3RZChiSquaredCuts(modules, lowerModuleIndex, middleModuleIndex, upperModuleIndex, rzChiSquared))
        return false;
    } else {
      rzChiSquared = -1;
    }

    rPhiChiSquared = computePT3RPhiChiSquared(acc, modules, lowerModuleIndices, pixelG, pixelF, pixelRadiusPCA, xs, ys);

    if (runChiSquaredCuts and pixelSegmentPt < 5.0f) {
      if (not passPT3RPhiChiSquaredCuts(modules, lowerModuleIndex, middleModuleIndex, upperModuleIndex, rPhiChiSquared))
        return false;
    }

    float xPix[Params_pLS::kLayers] = {mds.anchorX()[pixelInnerMDIndex], mds.anchorX()[pixelOuterMDIndex]};
    float yPix[Params_pLS::kLayers] = {mds.anchorY()[pixelInnerMDIndex], mds.anchorY()[pixelOuterMDIndex]};
    rPhiChiSquaredInwards = computePT3RPhiChiSquaredInwards(g, f, tripletRadius, xPix, yPix);

    if (runChiSquaredCuts and pixelSegmentPt < 5.0f) {
      if (not passPT3RPhiChiSquaredInwardsCuts(
              modules, lowerModuleIndex, middleModuleIndex, upperModuleIndex, rPhiChiSquaredInwards))
        return false;
    }
    centerX = 0;
    centerY = 0;
    return true;
  }

  struct CreatePixelTripletsFromMap {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  ModulesConst modules,
                                  ModulesPixelConst modulesPixel,
                                  ObjectRangesConst ranges,
                                  MiniDoubletsConst mds,
                                  SegmentsConst segments,
                                  SegmentsPixelConst segmentsPixel,
                                  Triplets triplets,
                                  TripletsOccupancyConst tripletsOccupancy,
                                  PixelTriplets pixelTriplets,
                                  unsigned int* connectedPixelSize,
                                  unsigned int* connectedPixelIndex,
                                  unsigned int nPixelSegments) const {
      auto const globalBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridBlockExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (unsigned int i_pLS = globalThreadIdx[1]; i_pLS < nPixelSegments; i_pLS += gridThreadExtent[1]) {
        auto iLSModule_max = connectedPixelIndex[i_pLS] + connectedPixelSize[i_pLS];

        for (unsigned int iLSModule = connectedPixelIndex[i_pLS] + globalBlockIdx[0]; iLSModule < iLSModule_max;
             iLSModule += gridBlockExtent[0]) {
          uint16_t tripletLowerModuleIndex =
              modulesPixel.connectedPixels()
                  [iLSModule];  //connected pixels will have the appropriate lower module index by default!
#ifdef WARNINGS
          if (tripletLowerModuleIndex >= modules.nLowerModules()) {
            printf("tripletLowerModuleIndex %d >= modules.nLowerModules %d \n",
                   tripletLowerModuleIndex,
                   modules.nLowerModules());
            continue;  //sanity check
          }
#endif
          //Removes 2S-2S :FIXME: filter these out in the pixel map
          if (modules.moduleType()[tripletLowerModuleIndex] == TwoS)
            continue;

          uint16_t pixelModuleIndex = modules.nLowerModules();
          unsigned int nOuterTriplets = tripletsOccupancy.nTriplets()[tripletLowerModuleIndex];
          if (nOuterTriplets == 0)
            continue;

          unsigned int pixelSegmentIndex = ranges.segmentModuleIndices()[pixelModuleIndex] + i_pLS;

          if (segmentsPixel.isDup()[i_pLS])
            continue;
          if (segmentsPixel.partOfPT5()[i_pLS])
            continue;  //don't make pT3s for those pixels that are part of pT5

          short layer2_adjustment;
          if (modules.layers()[tripletLowerModuleIndex] == 1) {
            layer2_adjustment = 1;
          }  //get upper segment to be in second layer
          else if (modules.layers()[tripletLowerModuleIndex] == 2) {
            layer2_adjustment = 0;
          }  // get lower segment to be in second layer
          else {
            continue;
          }

          //fetch the triplet
          for (unsigned int outerTripletArrayIndex = globalThreadIdx[2]; outerTripletArrayIndex < nOuterTriplets;
               outerTripletArrayIndex += gridThreadExtent[2]) {
            unsigned int outerTripletIndex =
                ranges.tripletModuleIndices()[tripletLowerModuleIndex] + outerTripletArrayIndex;
            if (modules.moduleType()[triplets.lowerModuleIndices()[outerTripletIndex][1]] == TwoS)
              continue;  //REMOVES PS-2S

            if (triplets.partOfPT5()[outerTripletIndex])
              continue;  //don't create pT3s for T3s accounted in pT5s

            float pixelRadius, tripletRadius, rPhiChiSquared, rzChiSquared, rPhiChiSquaredInwards, centerX, centerY;
            bool success = runPixelTripletDefaultAlgo(acc,
                                                      modules,
                                                      ranges,
                                                      mds,
                                                      segments,
                                                      segmentsPixel,
                                                      triplets,
                                                      pixelSegmentIndex,
                                                      outerTripletIndex,
                                                      pixelRadius,
                                                      tripletRadius,
                                                      centerX,
                                                      centerY,
                                                      rzChiSquared,
                                                      rPhiChiSquared,
                                                      rPhiChiSquaredInwards);

            if (success) {
              float phi =
                  mds.anchorPhi()[segments
                                      .mdIndices()[triplets.segmentIndices()[outerTripletIndex][0]][layer2_adjustment]];
              float eta =
                  mds.anchorEta()[segments
                                      .mdIndices()[triplets.segmentIndices()[outerTripletIndex][0]][layer2_adjustment]];
              float eta_pix = segmentsPixel.eta()[i_pLS];
              float phi_pix = segmentsPixel.phi()[i_pLS];
              float pt = segmentsPixel.ptIn()[i_pLS];
              float score = rPhiChiSquared + rPhiChiSquaredInwards;
              unsigned int totOccupancyPixelTriplets =
                  alpaka::atomicAdd(acc, &pixelTriplets.totOccupancyPixelTriplets(), 1u, alpaka::hierarchy::Threads{});
              if (totOccupancyPixelTriplets >= n_max_pixel_triplets) {
#ifdef WARNINGS
                printf("Pixel Triplet excess alert!\n");
#endif
              } else {
                unsigned int pixelTripletIndex =
                    alpaka::atomicAdd(acc, &pixelTriplets.nPixelTriplets(), 1u, alpaka::hierarchy::Threads{});
                addPixelTripletToMemory(mds,
                                        segments,
                                        triplets,
                                        pixelTriplets,
                                        pixelSegmentIndex,
                                        outerTripletIndex,
                                        pixelRadius,
                                        tripletRadius,
                                        centerX,
                                        centerY,
                                        rPhiChiSquared,
                                        rPhiChiSquaredInwards,
                                        rzChiSquared,
                                        pixelTripletIndex,
                                        pt,
                                        eta,
                                        phi,
                                        eta_pix,
                                        phi_pix,
                                        score);
                triplets.partOfPT3()[outerTripletIndex] = true;
              }
            }
          }  // for outerTripletArrayIndex
        }  // for iLSModule < iLSModule_max
      }  // for i_pLS
    }
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runTripletDefaultAlgoPPBB(TAcc const& acc,
                                                                ModulesConst modules,
                                                                ObjectRangesConst ranges,
                                                                MiniDoubletsConst mds,
                                                                SegmentsConst segments,
                                                                SegmentsPixelConst segmentsPixel,
                                                                uint16_t pixelModuleIndex,
                                                                uint16_t outerInnerLowerModuleIndex,
                                                                uint16_t outerOuterLowerModuleIndex,
                                                                unsigned int innerSegmentIndex,
                                                                unsigned int outerSegmentIndex,
                                                                unsigned int firstMDIndex,
                                                                unsigned int secondMDIndex,
                                                                unsigned int thirdMDIndex,
                                                                unsigned int fourthMDIndex) {
    float dPhi, betaIn, betaOut, pt_beta, zLo, zHi, zLoPointed, zHiPointed, dPhiCut, betaOutCut;

    bool isPS_OutLo = (modules.moduleType()[outerInnerLowerModuleIndex] == PS);

    float rt_InLo = mds.anchorRt()[firstMDIndex];
    float rt_InUp = mds.anchorRt()[secondMDIndex];
    float rt_OutLo = mds.anchorRt()[thirdMDIndex];
    float rt_OutUp = mds.anchorRt()[fourthMDIndex];

    float z_InUp = mds.anchorZ()[secondMDIndex];
    float z_OutLo = mds.anchorZ()[thirdMDIndex];

    float x_InLo = mds.anchorX()[firstMDIndex];
    float x_InUp = mds.anchorX()[secondMDIndex];
    float x_OutLo = mds.anchorX()[thirdMDIndex];
    float x_OutUp = mds.anchorX()[fourthMDIndex];

    float y_InLo = mds.anchorY()[firstMDIndex];
    float y_InUp = mds.anchorY()[secondMDIndex];
    float y_OutLo = mds.anchorY()[thirdMDIndex];
    float y_OutUp = mds.anchorY()[fourthMDIndex];

    float rt_InOut = rt_InUp;

    if (alpaka::math::abs(acc, deltaPhi(acc, x_InUp, y_InUp, x_OutLo, y_OutLo)) > kPi / 2.f)
      return false;

    unsigned int pixelSegmentArrayIndex = innerSegmentIndex - ranges.segmentModuleIndices()[pixelModuleIndex];
    float ptIn = segmentsPixel.ptIn()[pixelSegmentArrayIndex];
    float ptSLo = ptIn;
    float px = segmentsPixel.px()[pixelSegmentArrayIndex];
    float py = segmentsPixel.py()[pixelSegmentArrayIndex];
    float pz = segmentsPixel.pz()[pixelSegmentArrayIndex];
    float ptErr = segmentsPixel.ptErr()[pixelSegmentArrayIndex];
    float etaErr = segmentsPixel.etaErr()[pixelSegmentArrayIndex];
    ptSLo = alpaka::math::max(acc, ptCut, ptSLo - 10.0f * alpaka::math::max(acc, ptErr, 0.005f * ptSLo));
    ptSLo = alpaka::math::min(acc, 10.0f, ptSLo);

    float alpha1GeV_OutLo =
        alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / ptCut, kSinAlphaMax));
    const float rtRatio_OutLoInOut =
        rt_OutLo / rt_InOut;  // Outer segment beginning rt divided by inner segment beginning rt;

    float dzDrtScale =
        alpaka::math::tan(acc, alpha1GeV_OutLo) / alpha1GeV_OutLo;  // The track can bend in r-z plane slightly
    const float zpitch_InLo = 0.05f;
    const float zpitch_InOut = 0.05f;
    float zpitch_OutLo = (isPS_OutLo ? kPixelPSZpitch : kStrip2SZpitch);
    float zGeom = zpitch_InLo + zpitch_OutLo;
    zHi = z_InUp + (z_InUp + kDeltaZLum) * (rtRatio_OutLoInOut - 1.f) * (z_InUp < 0.f ? 1.f : dzDrtScale) +
          (zpitch_InOut + zpitch_OutLo);
    zLo = z_InUp + (z_InUp - kDeltaZLum) * (rtRatio_OutLoInOut - 1.f) * (z_InUp > 0.f ? 1.f : dzDrtScale) -
          (zpitch_InOut + zpitch_OutLo);  //slope-correction only on outer end

    if ((z_OutLo < zLo) || (z_OutLo > zHi))
      return false;

    const float cosh2Eta = 1.f + (pz * pz) / (ptIn * ptIn);

    const float drt_OutLo_InUp = (rt_OutLo - rt_InUp);

    const float r3_InUp = alpaka::math::sqrt(acc, z_InUp * z_InUp + rt_InUp * rt_InUp);

    float drt_InSeg = rt_InOut - rt_InLo;

    const float thetaMuls2 =
        (kMulsInGeV * kMulsInGeV) * (0.1f + 0.2f * (rt_OutLo - rt_InUp) / 50.f) * (r3_InUp / rt_InUp);
    const float muls2 = thetaMuls2 * 9.f / (ptCut * ptCut) * 16.f;

    float dzErr = (drt_OutLo_InUp * drt_OutLo_InUp) * (etaErr * etaErr) * cosh2Eta;
    dzErr += 0.03f * 0.03f;  // Approximately account for IT module size
    dzErr *= 9.f;            // 3 sigma
    dzErr += muls2 * (drt_OutLo_InUp * drt_OutLo_InUp) / 3.f * cosh2Eta;
    dzErr += zGeom * zGeom;
    dzErr = alpaka::math::sqrt(acc, dzErr);

    const float dzDrIn = pz / ptIn;
    const float zWindow = dzErr / drt_InSeg * drt_OutLo_InUp + zGeom;
    const float dzMean = dzDrIn * drt_OutLo_InUp *
                         (1.f + drt_OutLo_InUp * drt_OutLo_InUp * 4 * k2Rinv1GeVf * k2Rinv1GeVf / ptIn / ptIn /
                                    24.f);  // with curved path correction
    // Constructing upper and lower bound
    zLoPointed = z_InUp + dzMean - zWindow;
    zHiPointed = z_InUp + dzMean + zWindow;

    if ((z_OutLo < zLoPointed) || (z_OutLo > zHiPointed))
      return false;

    const float pvOffset = 0.1f / rt_OutLo;
    dPhiCut = alpha1GeV_OutLo + alpaka::math::sqrt(acc, muls2 + pvOffset * pvOffset);

    //no dphipos cut
    float midPointX = 0.5f * (x_InLo + x_OutLo);
    float midPointY = 0.5f * (y_InLo + y_OutLo);

    float diffX = x_OutLo - x_InLo;
    float diffY = y_OutLo - y_InLo;

    dPhi = deltaPhi(acc, midPointX, midPointY, diffX, diffY);

    if (alpaka::math::abs(acc, dPhi) > dPhiCut)
      return false;

    //lots of array accesses below this...

    float alpha_InLo = __H2F(segments.dPhiChanges()[innerSegmentIndex]);
    float alpha_OutLo = __H2F(segments.dPhiChanges()[outerSegmentIndex]);

    bool isEC_lastLayer = modules.subdets()[outerOuterLowerModuleIndex] == Endcap and
                          modules.moduleType()[outerOuterLowerModuleIndex] == TwoS;

    float alpha_OutUp, alpha_OutUp_highEdge, alpha_OutUp_lowEdge;
    alpha_OutUp = deltaPhi(acc, x_OutUp, y_OutUp, x_OutUp - x_OutLo, y_OutUp - y_OutLo);

    alpha_OutUp_highEdge = alpha_OutUp;
    alpha_OutUp_lowEdge = alpha_OutUp;

    float tl_axis_x = x_OutUp - x_InUp;
    float tl_axis_y = y_OutUp - y_InUp;

    float tl_axis_highEdge_x = tl_axis_x;
    float tl_axis_highEdge_y = tl_axis_y;

    float tl_axis_lowEdge_x = tl_axis_x;
    float tl_axis_lowEdge_y = tl_axis_y;

    betaIn = -deltaPhi(acc, px, py, tl_axis_x, tl_axis_y);
    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;

    betaOut = -alpha_OutUp + deltaPhi(acc, x_OutUp, y_OutUp, tl_axis_x, tl_axis_y);

    float betaOutRHmin = betaOut;
    float betaOutRHmax = betaOut;

    if (isEC_lastLayer) {
      alpha_OutUp_highEdge = deltaPhi(acc,
                                      mds.anchorHighEdgeX()[fourthMDIndex],
                                      mds.anchorHighEdgeY()[fourthMDIndex],
                                      mds.anchorHighEdgeX()[fourthMDIndex] - x_OutLo,
                                      mds.anchorHighEdgeY()[fourthMDIndex] - y_OutLo);
      alpha_OutUp_lowEdge = deltaPhi(acc,
                                     mds.anchorLowEdgeX()[fourthMDIndex],
                                     mds.anchorLowEdgeY()[fourthMDIndex],
                                     mds.anchorLowEdgeX()[fourthMDIndex] - x_OutLo,
                                     mds.anchorLowEdgeY()[fourthMDIndex] - y_OutLo);

      tl_axis_highEdge_x = mds.anchorHighEdgeX()[fourthMDIndex] - x_InUp;
      tl_axis_highEdge_y = mds.anchorHighEdgeY()[fourthMDIndex] - y_InUp;
      tl_axis_lowEdge_x = mds.anchorLowEdgeX()[fourthMDIndex] - x_InUp;
      tl_axis_lowEdge_y = mds.anchorLowEdgeY()[fourthMDIndex] - y_InUp;

      betaOutRHmin = -alpha_OutUp_highEdge + deltaPhi(acc,
                                                      mds.anchorHighEdgeX()[fourthMDIndex],
                                                      mds.anchorHighEdgeY()[fourthMDIndex],
                                                      tl_axis_highEdge_x,
                                                      tl_axis_highEdge_y);
      betaOutRHmax = -alpha_OutUp_lowEdge + deltaPhi(acc,
                                                     mds.anchorLowEdgeX()[fourthMDIndex],
                                                     mds.anchorLowEdgeY()[fourthMDIndex],
                                                     tl_axis_lowEdge_x,
                                                     tl_axis_lowEdge_y);
    }

    //beta computation
    float drt_tl_axis = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);

    //innerOuterAnchor - innerInnerAnchor
    const float rt_InSeg =
        alpaka::math::sqrt(acc, (x_InUp - x_InLo) * (x_InUp - x_InLo) + (y_InUp - y_InLo) * (y_InUp - y_InLo));

    //no betaIn cut for the pixels
    float betaAv = 0.5f * (betaIn + betaOut);
    pt_beta = ptIn;

    int lIn = 0;
    int lOut = isEC_lastLayer ? 11 : 5;
    float sdOut_dr =
        alpaka::math::sqrt(acc, (x_OutUp - x_OutLo) * (x_OutUp - x_OutLo) + (y_OutUp - y_OutLo) * (y_OutUp - y_OutLo));
    float sdOut_d = rt_OutUp - rt_OutLo;

    runDeltaBetaIterations(acc, betaIn, betaOut, betaAv, pt_beta, rt_InSeg, sdOut_dr, drt_tl_axis, lIn);

    const float betaInMMSF = (alpaka::math::abs(acc, betaInRHmin + betaInRHmax) > 0)
                                 ? (2.f * betaIn / alpaka::math::abs(acc, betaInRHmin + betaInRHmax))
                                 : 0.;  //mean value of min,max is the old betaIn
    const float betaOutMMSF = (alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax) > 0)
                                  ? (2.f * betaOut / alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax))
                                  : 0.;
    betaInRHmin *= betaInMMSF;
    betaInRHmax *= betaInMMSF;
    betaOutRHmin *= betaOutMMSF;
    betaOutRHmax *= betaOutMMSF;

    float min_ptBeta_ptBetaMax = alpaka::math::min(
        acc, alpaka::math::abs(acc, pt_beta), kPt_betaMax);  //need to confirm the range-out value of 7 GeV
    const float dBetaMuls2 = thetaMuls2 * 16.f / (min_ptBeta_ptBetaMax * min_ptBeta_ptBetaMax);
    const float alphaInAbsReg =
        alpaka::math::max(acc,
                          alpaka::math::abs(acc, alpha_InLo),
                          alpaka::math::asin(acc, alpaka::math::min(acc, rt_InUp * k2Rinv1GeVf / 3.0f, kSinAlphaMax)));
    const float alphaOutAbsReg =
        alpaka::math::max(acc,
                          alpaka::math::abs(acc, alpha_OutLo),
                          alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / 3.0f, kSinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : alpaka::math::abs(acc, alphaInAbsReg * kDeltaZLum / z_InUp);
    const float dBetaOutLum = lOut < 11 ? 0.0f : alpaka::math::abs(acc, alphaOutAbsReg * kDeltaZLum / z_OutLo);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);

    const float sinDPhi = alpaka::math::sin(acc, dPhi);
    const float dBetaRIn2 = 0;  // TODO-RH

    float dBetaROut = 0;
    if (isEC_lastLayer) {
      dBetaROut = (alpaka::math::sqrt(acc,
                                      mds.anchorHighEdgeX()[fourthMDIndex] * mds.anchorHighEdgeX()[fourthMDIndex] +
                                          mds.anchorHighEdgeY()[fourthMDIndex] * mds.anchorHighEdgeY()[fourthMDIndex]) -
                   alpaka::math::sqrt(acc,
                                      mds.anchorLowEdgeX()[fourthMDIndex] * mds.anchorLowEdgeX()[fourthMDIndex] +
                                          mds.anchorLowEdgeY()[fourthMDIndex] * mds.anchorLowEdgeY()[fourthMDIndex])) *
                  sinDPhi / drt_tl_axis;
    }

    const float dBetaROut2 = dBetaROut * dBetaROut;

    //FIXME: need faster version
    betaOutCut = alpaka::math::asin(acc, alpaka::math::min(acc, drt_tl_axis * k2Rinv1GeVf / ptCut, kSinAlphaMax)) +
                 (0.02f / sdOut_d) + alpaka::math::sqrt(acc, dBetaLum2 + dBetaMuls2);

    //Cut #6: The real beta cut
    if (alpaka::math::abs(acc, betaOut) >= betaOutCut)
      return false;
    const float dBetaRes = 0.02f / alpaka::math::min(acc, sdOut_d, drt_InSeg);
    const float dBetaCut2 =
        (dBetaRes * dBetaRes * 2.0f + dBetaMuls2 + dBetaLum2 + dBetaRIn2 + dBetaROut2 +
         0.25f *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));
    float dBeta = betaIn - betaOut;
    return dBeta * dBeta <= dBetaCut2;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runTripletDefaultAlgoPPEE(TAcc const& acc,
                                                                ModulesConst modules,
                                                                ObjectRangesConst ranges,
                                                                MiniDoubletsConst mds,
                                                                SegmentsConst segments,
                                                                SegmentsPixelConst segmentsPixel,
                                                                uint16_t pixelModuleIndex,
                                                                uint16_t outerInnerLowerModuleIndex,
                                                                uint16_t outerOuterLowerModuleIndex,
                                                                unsigned int innerSegmentIndex,
                                                                unsigned int outerSegmentIndex,
                                                                unsigned int firstMDIndex,
                                                                unsigned int secondMDIndex,
                                                                unsigned int thirdMDIndex,
                                                                unsigned int fourthMDIndex) {
    float dPhi, betaIn, betaOut, pt_beta, rtLo, rtHi, dPhiCut, betaOutCut;

    bool isPS_OutLo = (modules.moduleType()[outerInnerLowerModuleIndex] == PS);

    float z_InUp = mds.anchorZ()[secondMDIndex];
    float z_OutLo = mds.anchorZ()[thirdMDIndex];

    if (z_InUp * z_OutLo <= 0)
      return false;

    float rt_InLo = mds.anchorRt()[firstMDIndex];
    float rt_InUp = mds.anchorRt()[secondMDIndex];
    float rt_OutLo = mds.anchorRt()[thirdMDIndex];
    float rt_OutUp = mds.anchorRt()[fourthMDIndex];

    float x_InLo = mds.anchorX()[firstMDIndex];
    float x_InUp = mds.anchorX()[secondMDIndex];
    float x_OutLo = mds.anchorX()[thirdMDIndex];
    float x_OutUp = mds.anchorX()[fourthMDIndex];

    float y_InLo = mds.anchorY()[firstMDIndex];
    float y_InUp = mds.anchorY()[secondMDIndex];
    float y_OutLo = mds.anchorY()[thirdMDIndex];
    float y_OutUp = mds.anchorY()[fourthMDIndex];

    unsigned int pixelSegmentArrayIndex = innerSegmentIndex - ranges.segmentModuleIndices()[pixelModuleIndex];

    float ptIn = segmentsPixel.ptIn()[pixelSegmentArrayIndex];
    float ptSLo = ptIn;
    float px = segmentsPixel.px()[pixelSegmentArrayIndex];
    float py = segmentsPixel.py()[pixelSegmentArrayIndex];
    float pz = segmentsPixel.pz()[pixelSegmentArrayIndex];
    float ptErr = segmentsPixel.ptErr()[pixelSegmentArrayIndex];
    float etaErr = segmentsPixel.etaErr()[pixelSegmentArrayIndex];

    ptSLo = alpaka::math::max(acc, ptCut, ptSLo - 10.0f * alpaka::math::max(acc, ptErr, 0.005f * ptSLo));
    ptSLo = alpaka::math::min(acc, 10.0f, ptSLo);

    const float zpitch_InLo = 0.05f;
    float zpitch_OutLo = (isPS_OutLo ? kPixelPSZpitch : kStrip2SZpitch);
    float zGeom = zpitch_InLo + zpitch_OutLo;

    const float slope = alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / ptCut, kSinAlphaMax));
    const float dzDrtScale = alpaka::math::tan(acc, slope) / slope;  //FIXME: need approximate value

    const float dLum = alpaka::math::copysign(acc, kDeltaZLum, z_InUp);
    bool isOutSgInnerMDPS = modules.moduleType()[outerInnerLowerModuleIndex] == PS;

    const float rtGeom1 = isOutSgInnerMDPS
                              ? kPixelPSZpitch
                              : kStrip2SZpitch;  //FIXME: make this chosen by configuration for lay11,12 full PS
    const float zGeom1 = alpaka::math::copysign(acc, zGeom, z_InUp);  //used in B-E region
    rtLo = rt_InUp * (1.f + (z_OutLo - z_InUp - zGeom1) / (z_InUp + zGeom1 + dLum) / dzDrtScale) -
           rtGeom1;  //slope correction only on the lower end

    float zInForHi = z_InUp - zGeom1 - dLum;
    if (zInForHi * z_InUp < 0)
      zInForHi = alpaka::math::copysign(acc, 0.1f, z_InUp);
    rtHi = rt_InUp * (1.f + (z_OutLo - z_InUp + zGeom1) / zInForHi) + rtGeom1;

    // Cut #2: rt condition
    if ((rt_OutLo < rtLo) || (rt_OutLo > rtHi))
      return false;

    const float dzOutInAbs = alpaka::math::abs(acc, z_OutLo - z_InUp);
    const float cosh2Eta = 1.f + (pz * pz) / (ptIn * ptIn);
    const float multDzDr2 = (dzOutInAbs * dzOutInAbs) * cosh2Eta / ((cosh2Eta - 1.f) * (cosh2Eta - 1.f));
    const float r3_InUp = alpaka::math::sqrt(acc, z_InUp * z_InUp + rt_InUp * rt_InUp);
    const float thetaMuls2 =
        (kMulsInGeV * kMulsInGeV) * (0.1f + 0.2f * (rt_OutLo - rt_InUp) / 50.f) * (r3_InUp / rt_InUp);
    const float muls2 = thetaMuls2 * 9.f / (ptCut * ptCut) * 16.f;

    float drtErr = (etaErr * etaErr) * multDzDr2;
    drtErr += 0.03f * 0.03f;  // Approximately account for IT module size
    drtErr *= 9.f;            // 3 sigma
    drtErr += muls2 * multDzDr2 / 3.f * cosh2Eta;
    drtErr = alpaka::math::sqrt(acc, drtErr);
    const float drtDzIn = alpaka::math::abs(acc, ptIn / pz);

    const float drt_OutLo_InUp = (rt_OutLo - rt_InUp);  // drOutIn

    const float rtWindow = drtErr + rtGeom1;
    const float drtMean = drtDzIn * dzOutInAbs *
                          (1.f - drt_OutLo_InUp * drt_OutLo_InUp * 4 * k2Rinv1GeVf * k2Rinv1GeVf / ptIn / ptIn /
                                     24.f);  // with curved path correction
    const float rtLo_point = rt_InUp + drtMean - rtWindow;
    const float rtHi_point = rt_InUp + drtMean + rtWindow;

    // Cut #3: rt-z pointed
    if ((rt_OutLo < rtLo_point) || (rt_OutLo > rtHi_point))
      return false;

    const float alpha1GeV_OutLo =
        alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / ptCut, kSinAlphaMax));
    const float pvOffset = 0.1f / rt_OutLo;
    dPhiCut = alpha1GeV_OutLo + alpaka::math::sqrt(acc, muls2 + pvOffset * pvOffset);

    float midPointX = 0.5f * (x_InLo + x_OutLo);
    float midPointY = 0.5f * (y_InLo + y_OutLo);

    float diffX = x_OutLo - x_InLo;
    float diffY = y_OutLo - y_InLo;

    dPhi = deltaPhi(acc, midPointX, midPointY, diffX, diffY);

    // Cut #5: deltaPhiChange
    if (alpaka::math::abs(acc, dPhi) > dPhiCut)
      return false;

    float alpha_InLo = __H2F(segments.dPhiChanges()[innerSegmentIndex]);
    float alpha_OutLo = __H2F(segments.dPhiChanges()[outerSegmentIndex]);

    bool isEC_lastLayer = modules.subdets()[outerOuterLowerModuleIndex] == Endcap and
                          modules.moduleType()[outerOuterLowerModuleIndex] == TwoS;

    float alpha_OutUp, alpha_OutUp_highEdge, alpha_OutUp_lowEdge;

    alpha_OutUp = deltaPhi(acc, x_OutUp, y_OutUp, x_OutUp - x_OutLo, y_OutUp - y_OutLo);
    alpha_OutUp_highEdge = alpha_OutUp;
    alpha_OutUp_lowEdge = alpha_OutUp;

    float tl_axis_x = x_OutUp - x_InUp;
    float tl_axis_y = y_OutUp - y_InUp;

    float tl_axis_highEdge_x = tl_axis_x;
    float tl_axis_highEdge_y = tl_axis_y;

    float tl_axis_lowEdge_x = tl_axis_x;
    float tl_axis_lowEdge_y = tl_axis_y;

    betaIn = -deltaPhi(acc, px, py, tl_axis_x, tl_axis_y);
    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;

    betaOut = -alpha_OutUp + deltaPhi(acc, x_OutUp, y_OutUp, tl_axis_x, tl_axis_y);
    float betaOutRHmin = betaOut;
    float betaOutRHmax = betaOut;

    if (isEC_lastLayer) {
      alpha_OutUp_highEdge = deltaPhi(acc,
                                      mds.anchorHighEdgeX()[fourthMDIndex],
                                      mds.anchorHighEdgeY()[fourthMDIndex],
                                      mds.anchorHighEdgeX()[fourthMDIndex] - x_OutLo,
                                      mds.anchorHighEdgeY()[fourthMDIndex] - y_OutLo);
      alpha_OutUp_lowEdge = deltaPhi(acc,
                                     mds.anchorLowEdgeX()[fourthMDIndex],
                                     mds.anchorLowEdgeY()[fourthMDIndex],
                                     mds.anchorLowEdgeX()[fourthMDIndex] - x_OutLo,
                                     mds.anchorLowEdgeY()[fourthMDIndex] - y_OutLo);

      tl_axis_highEdge_x = mds.anchorHighEdgeX()[fourthMDIndex] - x_InUp;
      tl_axis_highEdge_y = mds.anchorHighEdgeY()[fourthMDIndex] - y_InUp;
      tl_axis_lowEdge_x = mds.anchorLowEdgeX()[fourthMDIndex] - x_InUp;
      tl_axis_lowEdge_y = mds.anchorLowEdgeY()[fourthMDIndex] - y_InUp;

      betaOutRHmin = -alpha_OutUp_highEdge + deltaPhi(acc,
                                                      mds.anchorHighEdgeX()[fourthMDIndex],
                                                      mds.anchorHighEdgeY()[fourthMDIndex],
                                                      tl_axis_highEdge_x,
                                                      tl_axis_highEdge_y);
      betaOutRHmax = -alpha_OutUp_lowEdge + deltaPhi(acc,
                                                     mds.anchorLowEdgeX()[fourthMDIndex],
                                                     mds.anchorLowEdgeY()[fourthMDIndex],
                                                     tl_axis_lowEdge_x,
                                                     tl_axis_lowEdge_y);
    }

    //beta computation
    float drt_tl_axis = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
    //no betaIn cut for the pixels
    const float rt_InSeg =
        alpaka::math::sqrt(acc, (x_InUp - x_InLo) * (x_InUp - x_InLo) + (y_InUp - y_InLo) * (y_InUp - y_InLo));

    float betaAv = 0.5f * (betaIn + betaOut);
    pt_beta = ptIn;

    int lIn = 0;
    int lOut = isEC_lastLayer ? 11 : 5;
    float sdOut_dr =
        alpaka::math::sqrt(acc, (x_OutUp - x_OutLo) * (x_OutUp - x_OutLo) + (y_OutUp - y_OutLo) * (y_OutUp - y_OutLo));
    float sdOut_d = rt_OutUp - rt_OutLo;

    runDeltaBetaIterations(acc, betaIn, betaOut, betaAv, pt_beta, rt_InSeg, sdOut_dr, drt_tl_axis, lIn);

    const float betaInMMSF = (alpaka::math::abs(acc, betaInRHmin + betaInRHmax) > 0)
                                 ? (2.f * betaIn / alpaka::math::abs(acc, betaInRHmin + betaInRHmax))
                                 : 0.;  //mean value of min,max is the old betaIn
    const float betaOutMMSF = (alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax) > 0)
                                  ? (2.f * betaOut / alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax))
                                  : 0.;
    betaInRHmin *= betaInMMSF;
    betaInRHmax *= betaInMMSF;
    betaOutRHmin *= betaOutMMSF;
    betaOutRHmax *= betaOutMMSF;

    float min_ptBeta_ptBetaMax = alpaka::math::min(
        acc, alpaka::math::abs(acc, pt_beta), kPt_betaMax);  //need to confirm the range-out value of 7 GeV
    const float dBetaMuls2 = thetaMuls2 * 16.f / (min_ptBeta_ptBetaMax * min_ptBeta_ptBetaMax);

    const float alphaInAbsReg =
        alpaka::math::max(acc,
                          alpaka::math::abs(acc, alpha_InLo),
                          alpaka::math::asin(acc, alpaka::math::min(acc, rt_InUp * k2Rinv1GeVf / 3.0f, kSinAlphaMax)));
    const float alphaOutAbsReg =
        alpaka::math::max(acc,
                          alpaka::math::abs(acc, alpha_OutLo),
                          alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / 3.0f, kSinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : alpaka::math::abs(acc, alphaInAbsReg * kDeltaZLum / z_InUp);
    const float dBetaOutLum = lOut < 11 ? 0.0f : alpaka::math::abs(acc, alphaOutAbsReg * kDeltaZLum / z_OutLo);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);

    const float sinDPhi = alpaka::math::sin(acc, dPhi);
    const float dBetaRIn2 = 0;  // TODO-RH

    float dBetaROut = 0;
    if (isEC_lastLayer) {
      dBetaROut = (alpaka::math::sqrt(acc,
                                      mds.anchorHighEdgeX()[fourthMDIndex] * mds.anchorHighEdgeX()[fourthMDIndex] +
                                          mds.anchorHighEdgeY()[fourthMDIndex] * mds.anchorHighEdgeY()[fourthMDIndex]) -
                   alpaka::math::sqrt(acc,
                                      mds.anchorLowEdgeX()[fourthMDIndex] * mds.anchorLowEdgeX()[fourthMDIndex] +
                                          mds.anchorLowEdgeY()[fourthMDIndex] * mds.anchorLowEdgeY()[fourthMDIndex])) *
                  sinDPhi / drt_tl_axis;
    }

    const float dBetaROut2 = dBetaROut * dBetaROut;

    betaOutCut =
        alpaka::math::asin(
            acc, alpaka::math::min(acc, drt_tl_axis * k2Rinv1GeVf / ptCut, kSinAlphaMax))  //FIXME: need faster version
        + (0.02f / sdOut_d) + alpaka::math::sqrt(acc, dBetaLum2 + dBetaMuls2);

    //Cut #6: The real beta cut
    if (alpaka::math::abs(acc, betaOut) >= betaOutCut)
      return false;

    float drt_InSeg = rt_InUp - rt_InLo;

    const float dBetaRes = 0.02f / alpaka::math::min(acc, sdOut_d, drt_InSeg);
    const float dBetaCut2 =
        (dBetaRes * dBetaRes * 2.0f + dBetaMuls2 + dBetaLum2 + dBetaRIn2 + dBetaROut2 +
         0.25f *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));
    float dBeta = betaIn - betaOut;
    return dBeta * dBeta <= dBetaCut2;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
