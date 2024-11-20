#ifndef RecoTracker_LSTCore_src_alpaka_Segment_h
#define RecoTracker_LSTCore_src_alpaka_Segment_h

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/SegmentsSoA.h"
#include "RecoTracker/LSTCore/interface/alpaka/SegmentsDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/EndcapGeometry.h"
#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"

#include "MiniDoublet.h"
#include "Hit.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool isTighterTiltedModules_seg(ModulesConst modules, unsigned int moduleIndex) {
    // The "tighter" tilted modules are the subset of tilted modules that have smaller spacing
    // This is the same as what was previously considered as"isNormalTiltedModules"
    // See Figure 9.1 of https://cds.cern.ch/record/2272264/files/CMS-TDR-014.pdf
    short subdet = modules.subdets()[moduleIndex];
    short layer = modules.layers()[moduleIndex];
    short side = modules.sides()[moduleIndex];
    short rod = modules.rods()[moduleIndex];

    return (subdet == Barrel) && (((side != Center) && (layer == 3)) ||
                                  ((side == NegZ) && (((layer == 2) && (rod > 5)) || ((layer == 1) && (rod > 9)))) ||
                                  ((side == PosZ) && (((layer == 2) && (rod < 8)) || ((layer == 1) && (rod < 4)))));
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool isTighterTiltedModules_seg(short subdet, short layer, short side, short rod) {
    // The "tighter" tilted modules are the subset of tilted modules that have smaller spacing
    // This is the same as what was previously considered as"isNormalTiltedModules"
    // See Figure 9.1 of https://cds.cern.ch/record/2272264/files/CMS-TDR-014.pdf
    return (subdet == Barrel) && (((side != Center) && (layer == 3)) ||
                                  ((side == NegZ) && (((layer == 2) && (rod > 5)) || ((layer == 1) && (rod > 9)))) ||
                                  ((side == PosZ) && (((layer == 2) && (rod < 8)) || ((layer == 1) && (rod < 4)))));
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float moduleGapSize_seg(short layer, short ring, short subdet, short side, short rod) {
    unsigned int iL = layer - 1;
    unsigned int iR = ring - 1;

    float moduleSeparation = 0;

    if (subdet == Barrel and side == Center) {
      moduleSeparation = kMiniDeltaFlat[iL];
    } else if (isTighterTiltedModules_seg(subdet, layer, side, rod)) {
      moduleSeparation = kMiniDeltaTilted[iL];
    } else if (subdet == Endcap) {
      moduleSeparation = kMiniDeltaEndcap[iL][iR];
    } else  //Loose tilted modules
    {
      moduleSeparation = kMiniDeltaLooseTilted[iL];
    }

    return moduleSeparation;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float moduleGapSize_seg(ModulesConst modules, unsigned int moduleIndex) {
    unsigned int iL = modules.layers()[moduleIndex] - 1;
    unsigned int iR = modules.rings()[moduleIndex] - 1;
    short subdet = modules.subdets()[moduleIndex];
    short side = modules.sides()[moduleIndex];

    float moduleSeparation = 0;

    if (subdet == Barrel and side == Center) {
      moduleSeparation = kMiniDeltaFlat[iL];
    } else if (isTighterTiltedModules_seg(modules, moduleIndex)) {
      moduleSeparation = kMiniDeltaTilted[iL];
    } else if (subdet == Endcap) {
      moduleSeparation = kMiniDeltaEndcap[iL][iR];
    } else  //Loose tilted modules
    {
      moduleSeparation = kMiniDeltaLooseTilted[iL];
    }

    return moduleSeparation;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void dAlphaThreshold(TAcc const& acc,
                                                      float* dAlphaThresholdValues,
                                                      ModulesConst modules,
                                                      MiniDoubletsConst mds,
                                                      float xIn,
                                                      float yIn,
                                                      float zIn,
                                                      float rtIn,
                                                      float xOut,
                                                      float yOut,
                                                      float zOut,
                                                      float rtOut,
                                                      uint16_t innerLowerModuleIndex,
                                                      uint16_t outerLowerModuleIndex,
                                                      unsigned int innerMDIndex,
                                                      unsigned int outerMDIndex) {
    float sdMuls = (modules.subdets()[innerLowerModuleIndex] == Barrel)
                       ? kMiniMulsPtScaleBarrel[modules.layers()[innerLowerModuleIndex] - 1] * 3.f / ptCut
                       : kMiniMulsPtScaleEndcap[modules.layers()[innerLowerModuleIndex] - 1] * 3.f / ptCut;

    //more accurate then outer rt - inner rt
    float segmentDr = alpaka::math::sqrt(acc, (yOut - yIn) * (yOut - yIn) + (xOut - xIn) * (xOut - xIn));

    const float dAlpha_Bfield =
        alpaka::math::asin(acc, alpaka::math::min(acc, segmentDr * k2Rinv1GeVf / ptCut, kSinAlphaMax));

    bool isInnerTilted =
        modules.subdets()[innerLowerModuleIndex] == Barrel and modules.sides()[innerLowerModuleIndex] != Center;
    bool isOuterTilted =
        modules.subdets()[outerLowerModuleIndex] == Barrel and modules.sides()[outerLowerModuleIndex] != Center;

    float drdzInner = modules.drdzs()[innerLowerModuleIndex];
    float drdzOuter = modules.drdzs()[outerLowerModuleIndex];
    float innerModuleGapSize = moduleGapSize_seg(modules, innerLowerModuleIndex);
    float outerModuleGapSize = moduleGapSize_seg(modules, outerLowerModuleIndex);
    const float innerminiTilt2 = isInnerTilted
                                     ? ((0.5f * 0.5f) * (kPixelPSZpitch * kPixelPSZpitch) * (drdzInner * drdzInner) /
                                        (1.f + drdzInner * drdzInner) / (innerModuleGapSize * innerModuleGapSize))
                                     : 0;

    const float outerminiTilt2 = isOuterTilted
                                     ? ((0.5f * 0.5f) * (kPixelPSZpitch * kPixelPSZpitch) * (drdzOuter * drdzOuter) /
                                        (1.f + drdzOuter * drdzOuter) / (outerModuleGapSize * outerModuleGapSize))
                                     : 0;

    float miniDelta = innerModuleGapSize;

    float sdLumForInnerMini2;
    float sdLumForOuterMini2;

    if (modules.subdets()[innerLowerModuleIndex] == Barrel) {
      sdLumForInnerMini2 = innerminiTilt2 * (dAlpha_Bfield * dAlpha_Bfield);
    } else {
      sdLumForInnerMini2 = (mds.dphis()[innerMDIndex] * mds.dphis()[innerMDIndex]) * (kDeltaZLum * kDeltaZLum) /
                           (mds.dzs()[innerMDIndex] * mds.dzs()[innerMDIndex]);
    }

    if (modules.subdets()[outerLowerModuleIndex] == Barrel) {
      sdLumForOuterMini2 = outerminiTilt2 * (dAlpha_Bfield * dAlpha_Bfield);
    } else {
      sdLumForOuterMini2 = (mds.dphis()[outerMDIndex] * mds.dphis()[outerMDIndex]) * (kDeltaZLum * kDeltaZLum) /
                           (mds.dzs()[outerMDIndex] * mds.dzs()[outerMDIndex]);
    }

    // Unique stuff for the segment dudes alone
    float dAlpha_res_inner =
        0.02f / miniDelta *
        (modules.subdets()[innerLowerModuleIndex] == Barrel ? 1.0f : alpaka::math::abs(acc, zIn) / rtIn);
    float dAlpha_res_outer =
        0.02f / miniDelta *
        (modules.subdets()[outerLowerModuleIndex] == Barrel ? 1.0f : alpaka::math::abs(acc, zOut) / rtOut);

    float dAlpha_res = dAlpha_res_inner + dAlpha_res_outer;

    if (modules.subdets()[innerLowerModuleIndex] == Barrel and modules.sides()[innerLowerModuleIndex] == Center) {
      dAlphaThresholdValues[0] = dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls);
    } else {
      dAlphaThresholdValues[0] =
          dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls + sdLumForInnerMini2);
    }

    if (modules.subdets()[outerLowerModuleIndex] == Barrel and modules.sides()[outerLowerModuleIndex] == Center) {
      dAlphaThresholdValues[1] = dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls);
    } else {
      dAlphaThresholdValues[1] =
          dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls + sdLumForOuterMini2);
    }

    //Inner to outer
    dAlphaThresholdValues[2] = dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls);
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addSegmentToMemory(Segments segments,
                                                         unsigned int lowerMDIndex,
                                                         unsigned int upperMDIndex,
                                                         uint16_t innerLowerModuleIndex,
                                                         uint16_t outerLowerModuleIndex,
                                                         unsigned int innerMDAnchorHitIndex,
                                                         unsigned int outerMDAnchorHitIndex,
                                                         float dPhi,
                                                         float dPhiMin,
                                                         float dPhiMax,
                                                         float dPhiChange,
                                                         float dPhiChangeMin,
                                                         float dPhiChangeMax,
                                                         unsigned int idx) {
    segments.mdIndices()[idx][0] = lowerMDIndex;
    segments.mdIndices()[idx][1] = upperMDIndex;
    segments.innerLowerModuleIndices()[idx] = innerLowerModuleIndex;
    segments.outerLowerModuleIndices()[idx] = outerLowerModuleIndex;
    segments.innerMiniDoubletAnchorHitIndices()[idx] = innerMDAnchorHitIndex;
    segments.outerMiniDoubletAnchorHitIndices()[idx] = outerMDAnchorHitIndex;

    segments.dPhis()[idx] = __F2H(dPhi);
    segments.dPhiMins()[idx] = __F2H(dPhiMin);
    segments.dPhiMaxs()[idx] = __F2H(dPhiMax);
    segments.dPhiChanges()[idx] = __F2H(dPhiChange);
    segments.dPhiChangeMins()[idx] = __F2H(dPhiChangeMin);
    segments.dPhiChangeMaxs()[idx] = __F2H(dPhiChangeMax);
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addPixelSegmentToMemory(TAcc const& acc,
                                                              Segments segments,
                                                              SegmentsPixel segmentsPixel,
                                                              MiniDoubletsConst mds,
                                                              unsigned int innerMDIndex,
                                                              unsigned int outerMDIndex,
                                                              uint16_t pixelModuleIndex,
                                                              unsigned int hitIdxs[4],
                                                              unsigned int innerAnchorHitIndex,
                                                              unsigned int outerAnchorHitIndex,
                                                              float dPhiChange,
                                                              unsigned int idx,
                                                              unsigned int pixelSegmentArrayIndex,
                                                              float score) {
    segments.mdIndices()[idx][0] = innerMDIndex;
    segments.mdIndices()[idx][1] = outerMDIndex;
    segments.innerLowerModuleIndices()[idx] = pixelModuleIndex;
    segments.outerLowerModuleIndices()[idx] = pixelModuleIndex;
    segments.innerMiniDoubletAnchorHitIndices()[idx] = innerAnchorHitIndex;
    segments.outerMiniDoubletAnchorHitIndices()[idx] = outerAnchorHitIndex;
    segments.dPhiChanges()[idx] = __F2H(dPhiChange);

    segmentsPixel.isDup()[pixelSegmentArrayIndex] = false;
    segmentsPixel.partOfPT5()[pixelSegmentArrayIndex] = false;
    segmentsPixel.score()[pixelSegmentArrayIndex] = score;
    segmentsPixel.pLSHitsIdxs()[pixelSegmentArrayIndex].x = hitIdxs[0];
    segmentsPixel.pLSHitsIdxs()[pixelSegmentArrayIndex].y = hitIdxs[1];
    segmentsPixel.pLSHitsIdxs()[pixelSegmentArrayIndex].z = hitIdxs[2];
    segmentsPixel.pLSHitsIdxs()[pixelSegmentArrayIndex].w = hitIdxs[3];

    //computing circle parameters
    /*
    The two anchor hits are r3PCA and r3LH. p3PCA pt, eta, phi is hitIndex1 x, y, z
    */
    float circleRadius = mds.outerX()[innerMDIndex] / (2 * k2Rinv1GeVf);
    float circlePhi = mds.outerZ()[innerMDIndex];
    float candidateCenterXs[] = {mds.anchorX()[innerMDIndex] + circleRadius * alpaka::math::sin(acc, circlePhi),
                                 mds.anchorX()[innerMDIndex] - circleRadius * alpaka::math::sin(acc, circlePhi)};
    float candidateCenterYs[] = {mds.anchorY()[innerMDIndex] - circleRadius * alpaka::math::cos(acc, circlePhi),
                                 mds.anchorY()[innerMDIndex] + circleRadius * alpaka::math::cos(acc, circlePhi)};

    //check which of the circles can accommodate r3LH better (we won't get perfect agreement)
    float bestChiSquared = kVerticalModuleSlope;
    float chiSquared;
    size_t bestIndex;
    for (size_t i = 0; i < 2; i++) {
      chiSquared = alpaka::math::abs(acc,
                                     alpaka::math::sqrt(acc,
                                                        (mds.anchorX()[outerMDIndex] - candidateCenterXs[i]) *
                                                                (mds.anchorX()[outerMDIndex] - candidateCenterXs[i]) +
                                                            (mds.anchorY()[outerMDIndex] - candidateCenterYs[i]) *
                                                                (mds.anchorY()[outerMDIndex] - candidateCenterYs[i])) -
                                         circleRadius);
      if (chiSquared < bestChiSquared) {
        bestChiSquared = chiSquared;
        bestIndex = i;
      }
    }
    segmentsPixel.circleCenterX()[pixelSegmentArrayIndex] = candidateCenterXs[bestIndex];
    segmentsPixel.circleCenterY()[pixelSegmentArrayIndex] = candidateCenterYs[bestIndex];
    segmentsPixel.circleRadius()[pixelSegmentArrayIndex] = circleRadius;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runSegmentDefaultAlgoBarrel(TAcc const& acc,
                                                                  ModulesConst modules,
                                                                  MiniDoubletsConst mds,
                                                                  uint16_t innerLowerModuleIndex,
                                                                  uint16_t outerLowerModuleIndex,
                                                                  unsigned int innerMDIndex,
                                                                  unsigned int outerMDIndex,
                                                                  float& dPhi,
                                                                  float& dPhiMin,
                                                                  float& dPhiMax,
                                                                  float& dPhiChange,
                                                                  float& dPhiChangeMin,
                                                                  float& dPhiChangeMax) {
    float sdMuls = (modules.subdets()[innerLowerModuleIndex] == Barrel)
                       ? kMiniMulsPtScaleBarrel[modules.layers()[innerLowerModuleIndex] - 1] * 3.f / ptCut
                       : kMiniMulsPtScaleEndcap[modules.layers()[innerLowerModuleIndex] - 1] * 3.f / ptCut;

    float xIn, yIn, zIn, rtIn, xOut, yOut, zOut, rtOut;

    xIn = mds.anchorX()[innerMDIndex];
    yIn = mds.anchorY()[innerMDIndex];
    zIn = mds.anchorZ()[innerMDIndex];
    rtIn = mds.anchorRt()[innerMDIndex];

    xOut = mds.anchorX()[outerMDIndex];
    yOut = mds.anchorY()[outerMDIndex];
    zOut = mds.anchorZ()[outerMDIndex];
    rtOut = mds.anchorRt()[outerMDIndex];

    float sdSlope = alpaka::math::asin(acc, alpaka::math::min(acc, rtOut * k2Rinv1GeVf / ptCut, kSinAlphaMax));
    float sdPVoff = 0.1f / rtOut;
    float dzDrtScale = alpaka::math::tan(acc, sdSlope) / sdSlope;  //FIXME: need appropriate value

    const float zGeom = modules.layers()[innerLowerModuleIndex] <= 2 ? 2.f * kPixelPSZpitch : 2.f * kStrip2SZpitch;

    float zLo = zIn + (zIn - kDeltaZLum) * (rtOut / rtIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) -
                zGeom;  //slope-correction only on outer end
    float zHi = zIn + (zIn + kDeltaZLum) * (rtOut / rtIn - 1.f) * (zIn < 0.f ? 1.f : dzDrtScale) + zGeom;

    if ((zOut < zLo) || (zOut > zHi))
      return false;

    float sdCut = sdSlope + alpaka::math::sqrt(acc, sdMuls * sdMuls + sdPVoff * sdPVoff);

    dPhi = phi_mpi_pi(acc, mds.anchorPhi()[outerMDIndex] - mds.anchorPhi()[innerMDIndex]);

    if (alpaka::math::abs(acc, dPhi) > sdCut)
      return false;

    dPhiChange = phi_mpi_pi(acc, phi(acc, xOut - xIn, yOut - yIn) - mds.anchorPhi()[innerMDIndex]);

    if (alpaka::math::abs(acc, dPhiChange) > sdCut)
      return false;

    float dAlphaThresholdValues[3];
    dAlphaThreshold(acc,
                    dAlphaThresholdValues,
                    modules,
                    mds,
                    xIn,
                    yIn,
                    zIn,
                    rtIn,
                    xOut,
                    yOut,
                    zOut,
                    rtOut,
                    innerLowerModuleIndex,
                    outerLowerModuleIndex,
                    innerMDIndex,
                    outerMDIndex);

    float innerMDAlpha = mds.dphichanges()[innerMDIndex];
    float outerMDAlpha = mds.dphichanges()[outerMDIndex];
    float dAlphaInnerMDSegment = innerMDAlpha - dPhiChange;
    float dAlphaOuterMDSegment = outerMDAlpha - dPhiChange;
    float dAlphaInnerMDOuterMD = innerMDAlpha - outerMDAlpha;

    float dAlphaInnerMDSegmentThreshold = dAlphaThresholdValues[0];
    float dAlphaOuterMDSegmentThreshold = dAlphaThresholdValues[1];
    float dAlphaInnerMDOuterMDThreshold = dAlphaThresholdValues[2];

    if (alpaka::math::abs(acc, dAlphaInnerMDSegment) >= dAlphaInnerMDSegmentThreshold)
      return false;
    if (alpaka::math::abs(acc, dAlphaOuterMDSegment) >= dAlphaOuterMDSegmentThreshold)
      return false;
    return alpaka::math::abs(acc, dAlphaInnerMDOuterMD) < dAlphaInnerMDOuterMDThreshold;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runSegmentDefaultAlgoEndcap(TAcc const& acc,
                                                                  ModulesConst modules,
                                                                  MiniDoubletsConst mds,
                                                                  uint16_t innerLowerModuleIndex,
                                                                  uint16_t outerLowerModuleIndex,
                                                                  unsigned int innerMDIndex,
                                                                  unsigned int outerMDIndex,
                                                                  float& dPhi,
                                                                  float& dPhiMin,
                                                                  float& dPhiMax,
                                                                  float& dPhiChange,
                                                                  float& dPhiChangeMin,
                                                                  float& dPhiChangeMax) {
    float xIn, yIn, zIn, rtIn, xOut, yOut, zOut, rtOut;

    xIn = mds.anchorX()[innerMDIndex];
    yIn = mds.anchorY()[innerMDIndex];
    zIn = mds.anchorZ()[innerMDIndex];
    rtIn = mds.anchorRt()[innerMDIndex];

    xOut = mds.anchorX()[outerMDIndex];
    yOut = mds.anchorY()[outerMDIndex];
    zOut = mds.anchorZ()[outerMDIndex];
    rtOut = mds.anchorRt()[outerMDIndex];

    bool outerLayerEndcapTwoS =
        (modules.subdets()[outerLowerModuleIndex] == Endcap) && (modules.moduleType()[outerLowerModuleIndex] == TwoS);

    float sdSlope = alpaka::math::asin(acc, alpaka::math::min(acc, rtOut * k2Rinv1GeVf / ptCut, kSinAlphaMax));
    float disks2SMinRadius = 60.f;

    float rtGeom = ((rtIn < disks2SMinRadius && rtOut < disks2SMinRadius)
                        ? (2.f * kPixelPSZpitch)
                        : ((rtIn < disks2SMinRadius || rtOut < disks2SMinRadius) ? (kPixelPSZpitch + kStrip2SZpitch)
                                                                                 : (2.f * kStrip2SZpitch)));

    //cut 0 - z compatibility
    if (zIn * zOut < 0)
      return false;

    float dz = zOut - zIn;
    float dLum = alpaka::math::copysign(acc, kDeltaZLum, zIn);
    float drtDzScale = sdSlope / alpaka::math::tan(acc, sdSlope);

    float rtLo = alpaka::math::max(
        acc, rtIn * (1.f + dz / (zIn + dLum) * drtDzScale) - rtGeom, rtIn - 0.5f * rtGeom);  //rt should increase
    float rtHi = rtIn * (zOut - dLum) / (zIn - dLum) +
                 rtGeom;  //dLum for luminous; rGeom for measurement size; no tanTheta_loc(pt) correction

    // Completeness
    if ((rtOut < rtLo) || (rtOut > rtHi))
      return false;

    dPhi = phi_mpi_pi(acc, mds.anchorPhi()[outerMDIndex] - mds.anchorPhi()[innerMDIndex]);

    float sdCut = sdSlope;
    if (outerLayerEndcapTwoS) {
      float dPhiPos_high = phi_mpi_pi(acc, mds.anchorHighEdgePhi()[outerMDIndex] - mds.anchorPhi()[innerMDIndex]);
      float dPhiPos_low = phi_mpi_pi(acc, mds.anchorLowEdgePhi()[outerMDIndex] - mds.anchorPhi()[innerMDIndex]);

      dPhiMax = alpaka::math::abs(acc, dPhiPos_high) > alpaka::math::abs(acc, dPhiPos_low) ? dPhiPos_high : dPhiPos_low;
      dPhiMin = alpaka::math::abs(acc, dPhiPos_high) > alpaka::math::abs(acc, dPhiPos_low) ? dPhiPos_low : dPhiPos_high;
    } else {
      dPhiMax = dPhi;
      dPhiMin = dPhi;
    }
    if (alpaka::math::abs(acc, dPhi) > sdCut)
      return false;

    float dzFrac = dz / zIn;
    dPhiChange = dPhi / dzFrac * (1.f + dzFrac);
    dPhiChangeMin = dPhiMin / dzFrac * (1.f + dzFrac);
    dPhiChangeMax = dPhiMax / dzFrac * (1.f + dzFrac);

    if (alpaka::math::abs(acc, dPhiChange) > sdCut)
      return false;

    float dAlphaThresholdValues[3];
    dAlphaThreshold(acc,
                    dAlphaThresholdValues,
                    modules,
                    mds,
                    xIn,
                    yIn,
                    zIn,
                    rtIn,
                    xOut,
                    yOut,
                    zOut,
                    rtOut,
                    innerLowerModuleIndex,
                    outerLowerModuleIndex,
                    innerMDIndex,
                    outerMDIndex);

    float innerMDAlpha = mds.dphichanges()[innerMDIndex];
    float outerMDAlpha = mds.dphichanges()[outerMDIndex];
    float dAlphaInnerMDSegment = innerMDAlpha - dPhiChange;
    float dAlphaOuterMDSegment = outerMDAlpha - dPhiChange;
    float dAlphaInnerMDOuterMD = innerMDAlpha - outerMDAlpha;

    float dAlphaInnerMDSegmentThreshold = dAlphaThresholdValues[0];
    float dAlphaOuterMDSegmentThreshold = dAlphaThresholdValues[1];
    float dAlphaInnerMDOuterMDThreshold = dAlphaThresholdValues[2];

    if (alpaka::math::abs(acc, dAlphaInnerMDSegment) >= dAlphaInnerMDSegmentThreshold)
      return false;
    if (alpaka::math::abs(acc, dAlphaOuterMDSegment) >= dAlphaOuterMDSegmentThreshold)
      return false;
    return alpaka::math::abs(acc, dAlphaInnerMDOuterMD) < dAlphaInnerMDOuterMDThreshold;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runSegmentDefaultAlgo(TAcc const& acc,
                                                            ModulesConst modules,
                                                            MiniDoubletsConst mds,
                                                            uint16_t innerLowerModuleIndex,
                                                            uint16_t outerLowerModuleIndex,
                                                            unsigned int innerMDIndex,
                                                            unsigned int outerMDIndex,
                                                            float& dPhi,
                                                            float& dPhiMin,
                                                            float& dPhiMax,
                                                            float& dPhiChange,
                                                            float& dPhiChangeMin,
                                                            float& dPhiChangeMax) {
    if (modules.subdets()[innerLowerModuleIndex] == Barrel and modules.subdets()[outerLowerModuleIndex] == Barrel) {
      return runSegmentDefaultAlgoBarrel(acc,
                                         modules,
                                         mds,
                                         innerLowerModuleIndex,
                                         outerLowerModuleIndex,
                                         innerMDIndex,
                                         outerMDIndex,
                                         dPhi,
                                         dPhiMin,
                                         dPhiMax,
                                         dPhiChange,
                                         dPhiChangeMin,
                                         dPhiChangeMax);
    } else {
      return runSegmentDefaultAlgoEndcap(acc,
                                         modules,
                                         mds,
                                         innerLowerModuleIndex,
                                         outerLowerModuleIndex,
                                         innerMDIndex,
                                         outerMDIndex,
                                         dPhi,
                                         dPhiMin,
                                         dPhiMax,
                                         dPhiChange,
                                         dPhiChangeMin,
                                         dPhiChangeMax);
    }
  }

  struct CreateSegments {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  ModulesConst modules,
                                  MiniDoubletsConst mds,
                                  MiniDoubletsOccupancyConst mdsOccupancy,
                                  Segments segments,
                                  SegmentsOccupancy segmentsOccupancy,
                                  ObjectRangesConst ranges) const {
      auto const globalBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
      auto const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
      auto const gridBlockExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc);
      auto const blockThreadExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);

      for (uint16_t innerLowerModuleIndex = globalBlockIdx[2]; innerLowerModuleIndex < modules.nLowerModules();
           innerLowerModuleIndex += gridBlockExtent[2]) {
        unsigned int nInnerMDs = mdsOccupancy.nMDs()[innerLowerModuleIndex];
        if (nInnerMDs == 0)
          continue;

        unsigned int nConnectedModules = modules.nConnectedModules()[innerLowerModuleIndex];

        for (uint16_t outerLowerModuleArrayIdx = blockThreadIdx[1]; outerLowerModuleArrayIdx < nConnectedModules;
             outerLowerModuleArrayIdx += blockThreadExtent[1]) {
          uint16_t outerLowerModuleIndex = modules.moduleMap()[innerLowerModuleIndex][outerLowerModuleArrayIdx];

          unsigned int nOuterMDs = mdsOccupancy.nMDs()[outerLowerModuleIndex];

          unsigned int limit = nInnerMDs * nOuterMDs;

          if (limit == 0)
            continue;
          for (unsigned int hitIndex = blockThreadIdx[2]; hitIndex < limit; hitIndex += blockThreadExtent[2]) {
            unsigned int innerMDArrayIdx = hitIndex / nOuterMDs;
            unsigned int outerMDArrayIdx = hitIndex % nOuterMDs;
            if (outerMDArrayIdx >= nOuterMDs)
              continue;

            unsigned int innerMDIndex = ranges.mdRanges()[innerLowerModuleIndex][0] + innerMDArrayIdx;
            unsigned int outerMDIndex = ranges.mdRanges()[outerLowerModuleIndex][0] + outerMDArrayIdx;

            float dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax;

            unsigned int innerMiniDoubletAnchorHitIndex = mds.anchorHitIndices()[innerMDIndex];
            unsigned int outerMiniDoubletAnchorHitIndex = mds.anchorHitIndices()[outerMDIndex];
            dPhiMin = 0;
            dPhiMax = 0;
            dPhiChangeMin = 0;
            dPhiChangeMax = 0;
            if (runSegmentDefaultAlgo(acc,
                                      modules,
                                      mds,
                                      innerLowerModuleIndex,
                                      outerLowerModuleIndex,
                                      innerMDIndex,
                                      outerMDIndex,
                                      dPhi,
                                      dPhiMin,
                                      dPhiMax,
                                      dPhiChange,
                                      dPhiChangeMin,
                                      dPhiChangeMax)) {
              unsigned int totOccupancySegments =
                  alpaka::atomicAdd(acc,
                                    &segmentsOccupancy.totOccupancySegments()[innerLowerModuleIndex],
                                    1u,
                                    alpaka::hierarchy::Threads{});
              if (static_cast<int>(totOccupancySegments) >= ranges.segmentModuleOccupancy()[innerLowerModuleIndex]) {
#ifdef WARNINGS
                printf("Segment excess alert! Module index = %d\n", innerLowerModuleIndex);
#endif
              } else {
                unsigned int segmentModuleIdx = alpaka::atomicAdd(
                    acc, &segmentsOccupancy.nSegments()[innerLowerModuleIndex], 1u, alpaka::hierarchy::Threads{});
                unsigned int segmentIdx = ranges.segmentModuleIndices()[innerLowerModuleIndex] + segmentModuleIdx;

                addSegmentToMemory(segments,
                                   innerMDIndex,
                                   outerMDIndex,
                                   innerLowerModuleIndex,
                                   outerLowerModuleIndex,
                                   innerMiniDoubletAnchorHitIndex,
                                   outerMiniDoubletAnchorHitIndex,
                                   dPhi,
                                   dPhiMin,
                                   dPhiMax,
                                   dPhiChange,
                                   dPhiChangeMin,
                                   dPhiChangeMax,
                                   segmentIdx);
              }
            }
          }
        }
      }
    }
  };

  struct CreateSegmentArrayRanges {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  ModulesConst modules,
                                  ObjectRanges ranges,
                                  MiniDoubletsConst mds) const {
      // implementation is 1D with a single block
      static_assert(std::is_same_v<TAcc, ALPAKA_ACCELERATOR_NAMESPACE::Acc1D>, "Should be Acc1D");
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      // Initialize variables in shared memory and set to 0
      int& nTotalSegments = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        nTotalSegments = 0;
      }
      alpaka::syncBlockThreads(acc);

      for (uint16_t i = globalThreadIdx[0]; i < modules.nLowerModules(); i += gridThreadExtent[0]) {
        if (modules.nConnectedModules()[i] == 0) {
          ranges.segmentModuleIndices()[i] = nTotalSegments;
          ranges.segmentModuleOccupancy()[i] = 0;
          continue;
        }

        short module_rings = modules.rings()[i];
        short module_layers = modules.layers()[i];
        short module_subdets = modules.subdets()[i];
        float module_eta = alpaka::math::abs(acc, modules.eta()[i]);

        int category_number;
        if (module_layers <= 3 && module_subdets == 5)
          category_number = 0;
        else if (module_layers >= 4 && module_subdets == 5)
          category_number = 1;
        else if (module_layers <= 2 && module_subdets == 4 && module_rings >= 11)
          category_number = 2;
        else if (module_layers >= 3 && module_subdets == 4 && module_rings >= 8)
          category_number = 2;
        else if (module_layers <= 2 && module_subdets == 4 && module_rings <= 10)
          category_number = 3;
        else if (module_layers >= 3 && module_subdets == 4 && module_rings <= 7)
          category_number = 3;
        else
          category_number = -1;

        int eta_number;
        if (module_eta < 0.75f)
          eta_number = 0;
        else if (module_eta < 1.5f)
          eta_number = 1;
        else if (module_eta < 2.25f)
          eta_number = 2;
        else if (module_eta < 3.0f)
          eta_number = 3;
        else
          eta_number = -1;

        int occupancy;
        if (category_number == 0 && eta_number == 0)
          occupancy = 572;
        else if (category_number == 0 && eta_number == 1)
          occupancy = 300;
        else if (category_number == 0 && eta_number == 2)
          occupancy = 183;
        else if (category_number == 0 && eta_number == 3)
          occupancy = 62;
        else if (category_number == 1 && eta_number == 0)
          occupancy = 191;
        else if (category_number == 1 && eta_number == 1)
          occupancy = 128;
        else if (category_number == 2 && eta_number == 1)
          occupancy = 107;
        else if (category_number == 2 && eta_number == 2)
          occupancy = 102;
        else if (category_number == 3 && eta_number == 1)
          occupancy = 64;
        else if (category_number == 3 && eta_number == 2)
          occupancy = 79;
        else if (category_number == 3 && eta_number == 3)
          occupancy = 85;
        else {
          occupancy = 0;
#ifdef WARNINGS
          printf("Unhandled case in createSegmentArrayRanges! Module index = %i\n", i);
#endif
        }

        int nTotSegs = alpaka::atomicAdd(acc, &nTotalSegments, occupancy, alpaka::hierarchy::Threads{});
        ranges.segmentModuleIndices()[i] = nTotSegs;
        ranges.segmentModuleOccupancy()[i] = occupancy;
      }

      // Wait for all threads to finish before reporting final values
      alpaka::syncBlockThreads(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        ranges.segmentModuleIndices()[modules.nLowerModules()] = nTotalSegments;
        ranges.nTotalSegs() = nTotalSegments;
      }
    }
  };

  struct AddSegmentRangesToEventExplicit {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  ModulesConst modules,
                                  SegmentsOccupancyConst segmentsOccupancy,
                                  ObjectRanges ranges) const {
      // implementation is 1D with a single block
      static_assert(std::is_same_v<TAcc, ALPAKA_ACCELERATOR_NAMESPACE::Acc1D>, "Should be Acc1D");
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (uint16_t i = globalThreadIdx[0]; i < modules.nLowerModules(); i += gridThreadExtent[0]) {
        if (segmentsOccupancy.nSegments()[i] == 0) {
          ranges.segmentRanges()[i][0] = -1;
          ranges.segmentRanges()[i][1] = -1;
        } else {
          ranges.segmentRanges()[i][0] = ranges.segmentModuleIndices()[i];
          ranges.segmentRanges()[i][1] = ranges.segmentModuleIndices()[i] + segmentsOccupancy.nSegments()[i] - 1;
        }
      }
    }
  };

  struct AddPixelSegmentToEventKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  ModulesConst modules,
                                  ObjectRangesConst ranges,
                                  HitsConst hits,
                                  MiniDoublets mds,
                                  Segments segments,
                                  SegmentsPixel segmentsPixel,
                                  unsigned int* hitIndices0,
                                  unsigned int* hitIndices1,
                                  unsigned int* hitIndices2,
                                  unsigned int* hitIndices3,
                                  float* dPhiChange,
                                  uint16_t pixelModuleIndex,
                                  int size) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (int tid = globalThreadIdx[2]; tid < size; tid += gridThreadExtent[2]) {
        unsigned int innerMDIndex = ranges.miniDoubletModuleIndices()[pixelModuleIndex] + 2 * (tid);
        unsigned int outerMDIndex = ranges.miniDoubletModuleIndices()[pixelModuleIndex] + 2 * (tid) + 1;
        unsigned int pixelSegmentIndex = ranges.segmentModuleIndices()[pixelModuleIndex] + tid;

        addMDToMemory(acc,
                      mds,
                      hits,
                      modules,
                      hitIndices0[tid],
                      hitIndices1[tid],
                      pixelModuleIndex,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      innerMDIndex);
        addMDToMemory(acc,
                      mds,
                      hits,
                      modules,
                      hitIndices2[tid],
                      hitIndices3[tid],
                      pixelModuleIndex,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      outerMDIndex);

        //in outer hits - pt, eta, phi
        float slope = alpaka::math::sinh(acc, hits.ys()[mds.outerHitIndices()[innerMDIndex]]);
        float intercept =
            hits.zs()[mds.anchorHitIndices()[innerMDIndex]] - slope * hits.rts()[mds.anchorHitIndices()[innerMDIndex]];
        float score_lsq = (hits.rts()[mds.anchorHitIndices()[outerMDIndex]] * slope + intercept) -
                          (hits.zs()[mds.anchorHitIndices()[outerMDIndex]]);
        score_lsq = score_lsq * score_lsq;

        unsigned int hits1[Params_pLS::kHits];
        hits1[0] = hits.idxs()[mds.anchorHitIndices()[innerMDIndex]];
        hits1[1] = hits.idxs()[mds.anchorHitIndices()[outerMDIndex]];
        hits1[2] = hits.idxs()[mds.outerHitIndices()[innerMDIndex]];
        hits1[3] = hits.idxs()[mds.outerHitIndices()[outerMDIndex]];
        addPixelSegmentToMemory(acc,
                                segments,
                                segmentsPixel,
                                mds,
                                innerMDIndex,
                                outerMDIndex,
                                pixelModuleIndex,
                                hits1,
                                hitIndices0[tid],
                                hitIndices2[tid],
                                dPhiChange[tid],
                                pixelSegmentIndex,
                                tid,
                                score_lsq);
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif
