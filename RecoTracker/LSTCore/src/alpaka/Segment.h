#ifndef RecoTracker_LSTCore_src_alpaka_Segment_h
#define RecoTracker_LSTCore_src_alpaka_Segment_h

#include <limits>

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/Utilities/interface/CMSUnrollLoop.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/SegmentsSoA.h"
#include "RecoTracker/LSTCore/interface/alpaka/SegmentsDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/PixelSegmentsSoA.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/HitsSoA.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsSoA.h"
#include "RecoTracker/LSTCore/interface/EndcapGeometry.h"
#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"

#include "NeuralNetwork.h"

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

  template <alpaka::concepts::Acc TAcc>
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
                                                      unsigned int outerMDIndex,
                                                      const float ptCut) {
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
                                                         float dPhiChange,
                                                         float dPhiChangeMin,
                                                         float dPhiChangeMax,
#ifdef CUT_VALUE_DEBUG
                                                         float dPhi,
                                                         float dPhiMin,
                                                         float dPhiMax,
                                                         float zHi,
                                                         float zLo,
                                                         float rtHi,
                                                         float rtLo,
                                                         float dAlphaInner,
                                                         float dAlphaOuter,
                                                         float dAlphaInnerOuter,
#endif
                                                         unsigned int idx) {
    segments.mdIndices()[idx][0] = lowerMDIndex;
    segments.mdIndices()[idx][1] = upperMDIndex;
    segments.innerLowerModuleIndices()[idx] = innerLowerModuleIndex;
    segments.outerLowerModuleIndices()[idx] = outerLowerModuleIndex;
    segments.innerMiniDoubletAnchorHitIndices()[idx] = innerMDAnchorHitIndex;
    segments.outerMiniDoubletAnchorHitIndices()[idx] = outerMDAnchorHitIndex;

    segments.dPhiChanges()[idx] = __F2H(dPhiChange);
#ifdef CUT_VALUE_DEBUG
    segments.dPhis()[idx] = __F2H(dPhi);
    segments.dPhiMins()[idx] = __F2H(dPhiMin);
    segments.dPhiMaxs()[idx] = __F2H(dPhiMax);
#endif
    segments.dPhiChangeMins()[idx] = __F2H(dPhiChangeMin);
    segments.dPhiChangeMaxs()[idx] = __F2H(dPhiChangeMax);

#ifdef CUT_VALUE_DEBUG
    segments.zHis()[idx] = __F2H(zHi);
    segments.zLos()[idx] = __F2H(zLo);
    segments.rtHis()[idx] = __F2H(rtHi);
    segments.rtLos()[idx] = __F2H(rtLo);
    segments.dAlphaInners()[idx] = __F2H(dAlphaInner);
    segments.dAlphaOuters()[idx] = __F2H(dAlphaOuter);
    segments.dAlphaInnerOuters()[idx] = __F2H(dAlphaInnerOuter);
#endif
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addPixelSegmentToMemory(TAcc const& acc,
                                                              Segments segments,
                                                              PixelSegments pixelSegments,
                                                              PixelSeedsConst pixelSeeds,
                                                              MiniDoubletsConst mds,
                                                              unsigned int innerMDIndex,
                                                              unsigned int outerMDIndex,
                                                              uint16_t pixelModuleIndex,
                                                              const Params_pLS::ArrayUxHits& hitIdxs,
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

    pixelSegments.isDup()[pixelSegmentArrayIndex] = false;
    pixelSegments.partOfPT5()[pixelSegmentArrayIndex] = false;
    pixelSegments.score()[pixelSegmentArrayIndex] = score;
    pixelSegments.pLSHitsIdxs()[pixelSegmentArrayIndex] = hitIdxs;

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
    float bestChiSquared = std::numeric_limits<float>::infinity();
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
    pixelSegments.circleCenterX()[pixelSegmentArrayIndex] = candidateCenterXs[bestIndex];
    pixelSegments.circleCenterY()[pixelSegmentArrayIndex] = candidateCenterYs[bestIndex];
    pixelSegments.circleRadius()[pixelSegmentArrayIndex] = circleRadius;

    float plsEmbed[Params_pLS::kEmbed];
    plsembdnn::runEmbed(acc,
                        pixelSeeds.eta()[pixelSegmentArrayIndex],
                        pixelSeeds.etaErr()[pixelSegmentArrayIndex],
                        pixelSeeds.phi()[pixelSegmentArrayIndex],
                        pixelSegments.circleCenterX()[pixelSegmentArrayIndex],
                        pixelSegments.circleCenterY()[pixelSegmentArrayIndex],
                        pixelSegments.circleRadius()[pixelSegmentArrayIndex],
                        pixelSeeds.ptIn()[pixelSegmentArrayIndex],
                        pixelSeeds.ptErr()[pixelSegmentArrayIndex],
                        static_cast<bool>(pixelSeeds.isQuad()[pixelSegmentArrayIndex]),
                        plsEmbed);

    CMS_UNROLL_LOOP for (unsigned k = 0; k < Params_pLS::kEmbed; ++k) {
      pixelSegments.plsEmbed()[pixelSegmentArrayIndex][k] = plsEmbed[k];
    }
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passDeltaPhiCutsBarrel(TAcc const& acc,
                                                             ModulesConst modules,
                                                             MiniDoubletsConst mds,
                                                             uint16_t innerLm,
                                                             uint16_t outerLm,
                                                             unsigned int innerMD,
                                                             unsigned int outerMD,
                                                             const float dPhi,
                                                             const float rtOut,
                                                             const float sdSlope,
                                                             const float ptCut) {
    const float sdMuls = kMiniMulsPtScaleBarrel[modules.layers()[innerLm] - 1] * 3.f / ptCut;
    const float sdPVoff = 0.1f / rtOut;
    const float sdCut = sdSlope + alpaka::math::sqrt(acc, sdMuls * sdMuls + sdPVoff * sdPVoff);

    if (alpaka::math::abs(acc, dPhi) > sdCut)
      return false;

    const float xIn = mds.anchorX()[innerMD];
    const float yIn = mds.anchorY()[innerMD];
    const float xOut = mds.anchorX()[outerMD];
    const float yOut = mds.anchorY()[outerMD];

    const float dPhiChange = cms::alpakatools::reducePhiRange(
        acc, cms::alpakatools::phi(acc, xOut - xIn, yOut - yIn) - mds.anchorPhi()[innerMD]);

    return alpaka::math::abs(acc, dPhiChange) < sdCut;
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passDeltaPhiCutsEndcap(TAcc const& acc,
                                                             ModulesConst modules,
                                                             MiniDoubletsConst mds,
                                                             uint16_t innerLm,
                                                             uint16_t outerLm,
                                                             unsigned int innerMD,
                                                             unsigned int outerMD,
                                                             const float dPhi,
                                                             const float rtOut,
                                                             const float sdSlope,
                                                             const float ptCut) {
    if (alpaka::math::abs(acc, dPhi) > sdSlope)
      return false;

    const float zIn = mds.anchorZ()[innerMD];
    const float zOut = mds.anchorZ()[outerMD];

    const float dz = zOut - zIn;
    const float dzFrac = dz / zIn;
    const float dPhiChange = dPhi / dzFrac * (1.f + dzFrac);

    return alpaka::math::abs(acc, dPhiChange) < sdSlope;
  }

  template <alpaka::concepts::Acc TAcc>
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
                                                                  float& dPhiChangeMax,
#ifdef CUT_VALUE_DEBUG
                                                                  float& dAlphaInnerMDSegment,
                                                                  float& dAlphaOuterMDSegment,
                                                                  float& dAlphaInnerMDOuterMD,
                                                                  float& zLo,
                                                                  float& zHi,
#endif
                                                                  const float ptCut) {
#ifndef CUT_VALUE_DEBUG
    float dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD;
    float zLo, zHi;
#endif
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
    float dzDrtScale = alpaka::math::tan(acc, sdSlope) / sdSlope;  //FIXME: need appropriate value

    const float zGeom = modules.layers()[innerLowerModuleIndex] <= 2 ? 2.f * kPixelPSZpitch : 2.f * kStrip2SZpitch;

    //slope-correction only on outer end
    zLo = zIn + (zIn - kDeltaZLum) * (rtOut / rtIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) - zGeom;
    zHi = zIn + (zIn + kDeltaZLum) * (rtOut / rtIn - 1.f) * (zIn < 0.f ? 1.f : dzDrtScale) + zGeom;

    if ((zOut < zLo) || (zOut > zHi))
      return false;

    dPhi = cms::alpakatools::reducePhiRange(acc, mds.anchorPhi()[outerMDIndex] - mds.anchorPhi()[innerMDIndex]);

    if (!passDeltaPhiCutsBarrel(acc,
                                modules,
                                mds,
                                innerLowerModuleIndex,
                                outerLowerModuleIndex,
                                innerMDIndex,
                                outerMDIndex,
                                dPhi,
                                rtOut,
                                sdSlope,
                                ptCut))
      return false;

    dPhiChange = cms::alpakatools::reducePhiRange(
        acc, cms::alpakatools::phi(acc, xOut - xIn, yOut - yIn) - mds.anchorPhi()[innerMDIndex]);

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
                    outerMDIndex,
                    ptCut);

    float innerMDAlpha = mds.dphichanges()[innerMDIndex];
    float outerMDAlpha = mds.dphichanges()[outerMDIndex];
    dAlphaInnerMDSegment = innerMDAlpha - dPhiChange;
    dAlphaOuterMDSegment = outerMDAlpha - dPhiChange;
    dAlphaInnerMDOuterMD = innerMDAlpha - outerMDAlpha;

    float dAlphaInnerMDSegmentThreshold = dAlphaThresholdValues[0];
    float dAlphaOuterMDSegmentThreshold = dAlphaThresholdValues[1];
    float dAlphaInnerMDOuterMDThreshold = dAlphaThresholdValues[2];

    if (alpaka::math::abs(acc, dAlphaInnerMDSegment) >= dAlphaInnerMDSegmentThreshold)
      return false;
    if (alpaka::math::abs(acc, dAlphaOuterMDSegment) >= dAlphaOuterMDSegmentThreshold)
      return false;
    return alpaka::math::abs(acc, dAlphaInnerMDOuterMD) < dAlphaInnerMDOuterMDThreshold;
  }

  template <alpaka::concepts::Acc TAcc>
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
                                                                  float& dPhiChangeMax,
#ifdef CUT_VALUE_DEBUG
                                                                  float& dAlphaInnerMDSegment,
                                                                  float& dAlphaOuterMDSegment,
                                                                  float& dAlphaInnerMDOuterMD,
                                                                  float& rtLo,
                                                                  float& rtHi,
#endif
                                                                  const float ptCut) {
#ifndef CUT_VALUE_DEBUG
    float dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD;
    float rtLo, rtHi;
#endif
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

    //rt should increase
    rtLo = alpaka::math::max(acc, rtIn * (1.f + dz / (zIn + dLum) * drtDzScale) - rtGeom, rtIn - 0.5f * rtGeom);
    //dLum for luminous; rGeom for measurement size; no tanTheta_loc(pt) correction
    rtHi = rtIn * (zOut - dLum) / (zIn - dLum) + rtGeom;

    // Completeness
    if ((rtOut < rtLo) || (rtOut > rtHi))
      return false;

    dPhi = cms::alpakatools::reducePhiRange(acc, mds.anchorPhi()[outerMDIndex] - mds.anchorPhi()[innerMDIndex]);

    if (!passDeltaPhiCutsEndcap(acc,
                                modules,
                                mds,
                                innerLowerModuleIndex,
                                outerLowerModuleIndex,
                                innerMDIndex,
                                outerMDIndex,
                                dPhi,
                                rtOut,
                                sdSlope,
                                ptCut))
      return false;

    if (outerLayerEndcapTwoS) {
      float dPhiPosHigh =
          cms::alpakatools::reducePhiRange(acc, mds.anchorHighEdgePhi()[outerMDIndex] - mds.anchorPhi()[innerMDIndex]);
      float dPhiPosLow =
          cms::alpakatools::reducePhiRange(acc, mds.anchorLowEdgePhi()[outerMDIndex] - mds.anchorPhi()[innerMDIndex]);

      dPhiMax = alpaka::math::abs(acc, dPhiPosHigh) > alpaka::math::abs(acc, dPhiPosLow) ? dPhiPosHigh : dPhiPosLow;
      dPhiMin = alpaka::math::abs(acc, dPhiPosHigh) > alpaka::math::abs(acc, dPhiPosLow) ? dPhiPosLow : dPhiPosHigh;
    } else {
      dPhiMax = dPhi;
      dPhiMin = dPhi;
    }

    float dzFrac = dz / zIn;
    dPhiChange = dPhi / dzFrac * (1.f + dzFrac);
    dPhiChangeMin = dPhiMin / dzFrac * (1.f + dzFrac);
    dPhiChangeMax = dPhiMax / dzFrac * (1.f + dzFrac);

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
                    outerMDIndex,
                    ptCut);

    float innerMDAlpha = mds.dphichanges()[innerMDIndex];
    float outerMDAlpha = mds.dphichanges()[outerMDIndex];
    dAlphaInnerMDSegment = innerMDAlpha - dPhiChange;
    dAlphaOuterMDSegment = outerMDAlpha - dPhiChange;
    dAlphaInnerMDOuterMD = innerMDAlpha - outerMDAlpha;

    float dAlphaInnerMDSegmentThreshold = dAlphaThresholdValues[0];
    float dAlphaOuterMDSegmentThreshold = dAlphaThresholdValues[1];
    float dAlphaInnerMDOuterMDThreshold = dAlphaThresholdValues[2];

    if (alpaka::math::abs(acc, dAlphaInnerMDSegment) >= dAlphaInnerMDSegmentThreshold)
      return false;
    if (alpaka::math::abs(acc, dAlphaOuterMDSegment) >= dAlphaOuterMDSegmentThreshold)
      return false;
    return alpaka::math::abs(acc, dAlphaInnerMDOuterMD) < dAlphaInnerMDOuterMDThreshold;
  }

  template <alpaka::concepts::Acc TAcc>
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
                                                            float& dPhiChangeMax,
#ifdef CUT_VALUE_DEBUG
                                                            float& dAlphaInnerMDSegment,
                                                            float& dAlphaOuterMDSegment,
                                                            float& dAlphaInnerMDOuterMD,
                                                            float& zLo,
                                                            float& zHi,
                                                            float& rtLo,
                                                            float& rtHi,
#endif
                                                            const float ptCut) {
    if (modules.subdets()[innerLowerModuleIndex] == Barrel and modules.subdets()[outerLowerModuleIndex] == Barrel) {
#ifdef CUT_VALUE_DEBUG
      rtLo = -999.f;
      rtHi = -999.f;
#endif
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
                                         dPhiChangeMax,
#ifdef CUT_VALUE_DEBUG
                                         dAlphaInnerMDSegment,
                                         dAlphaOuterMDSegment,
                                         dAlphaInnerMDOuterMD,
                                         zLo,
                                         zHi,
#endif
                                         ptCut);
    } else {
#ifdef CUT_VALUE_DEBUG
      zLo = -999.f;
      zHi = -999.f;
#endif
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
                                         dPhiChangeMax,
#ifdef CUT_VALUE_DEBUG
                                         dAlphaInnerMDSegment,
                                         dAlphaOuterMDSegment,
                                         dAlphaInnerMDOuterMD,
                                         rtLo,
                                         rtHi,
#endif
                                         ptCut);
    }
  }

  struct CreateSegments {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  ModulesConst modules,
                                  MiniDoubletsConst mds,
                                  MiniDoubletsOccupancyConst mdsOccupancy,
                                  Segments segments,
                                  SegmentsOccupancy segmentsOccupancy,
                                  ObjectRangesConst ranges,
                                  const float ptCut) const {
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[1] == 1) &&
                        (alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[2] == 1));
      for (uint16_t innerLowerModuleIndex : cms::alpakatools::uniform_groups_z(acc, modules.nLowerModules())) {
        unsigned int nInnerMDs = mdsOccupancy.nMDs()[innerLowerModuleIndex];
        if (nInnerMDs == 0)
          continue;

        unsigned int nConnectedModules = modules.nConnectedModules()[innerLowerModuleIndex];

        for (uint16_t outerLowerModuleArrayIdx : cms::alpakatools::uniform_elements_y(acc, nConnectedModules)) {
          uint16_t outerLowerModuleIndex = modules.moduleMap()[innerLowerModuleIndex][outerLowerModuleArrayIdx];

          unsigned int nOuterMDs = mdsOccupancy.nMDs()[outerLowerModuleIndex];

          unsigned int limit = nInnerMDs * nOuterMDs;

          if (limit == 0)
            continue;
          for (unsigned int hitIndex : cms::alpakatools::uniform_elements_x(acc, limit)) {
            unsigned int innerMDArrayIdx = hitIndex / nOuterMDs;
            unsigned int outerMDArrayIdx = hitIndex % nOuterMDs;

            unsigned int innerMDIndex = ranges.mdRanges()[innerLowerModuleIndex][0] + innerMDArrayIdx;
            unsigned int outerMDIndex = ranges.mdRanges()[outerLowerModuleIndex][0] + outerMDArrayIdx;

            float dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax;

#ifdef CUT_VALUE_DEBUG
            float zLo, zHi, rtLo, rtHi, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD;
#endif

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
                                      dPhiChangeMax,
#ifdef CUT_VALUE_DEBUG
                                      dAlphaInnerMDSegment,
                                      dAlphaOuterMDSegment,
                                      dAlphaInnerMDOuterMD,
                                      zLo,
                                      zHi,
                                      rtLo,
                                      rtHi,
#endif
                                      ptCut)) {
              unsigned int totOccupancySegments =
                  alpaka::atomicAdd(acc,
                                    &segmentsOccupancy.totOccupancySegments()[innerLowerModuleIndex],
                                    1u,
                                    alpaka::hierarchy::Threads{});
              if (static_cast<int>(totOccupancySegments) >= ranges.segmentModuleOccupancy()[innerLowerModuleIndex]) {
#ifdef WARNINGS
                printf("Segment excess alert! Module index = %d, Occupancy = %d\n",
                       innerLowerModuleIndex,
                       totOccupancySegments);
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
                                   dPhiChange,
                                   dPhiChangeMin,
                                   dPhiChangeMax,
#ifdef CUT_VALUE_DEBUG
                                   dPhi,
                                   dPhiMin,
                                   dPhiMax,
                                   zHi,
                                   zLo,
                                   rtHi,
                                   rtLo,
                                   dAlphaInnerMDSegment,
                                   dAlphaOuterMDSegment,
                                   dAlphaInnerMDOuterMD,
#endif
                                   segmentIdx);
              }
            }
          }
        }
      }
    }
  };

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passDeltaPhiCutsSelector(TAcc const& acc,
                                                               ModulesConst modules,
                                                               MiniDoubletsConst mds,
                                                               uint16_t innerLm,
                                                               uint16_t outerLm,
                                                               unsigned int innerMD,
                                                               unsigned int outerMD,
                                                               const float ptCut) {
    const float dPhi = cms::alpakatools::reducePhiRange(acc, mds.anchorPhi()[outerMD] - mds.anchorPhi()[innerMD]);
    const float rtOut = mds.anchorRt()[outerMD];
    const float sdSlope = alpaka::math::asin(acc, alpaka::math::min(acc, rtOut * k2Rinv1GeVf / ptCut, kSinAlphaMax));

    if (modules.subdets()[innerLm] == Barrel && modules.subdets()[outerLm] == Barrel)
      return passDeltaPhiCutsBarrel(acc, modules, mds, innerLm, outerLm, innerMD, outerMD, dPhi, rtOut, sdSlope, ptCut);
    else
      return passDeltaPhiCutsEndcap(acc, modules, mds, innerLm, outerLm, innerMD, outerMD, dPhi, rtOut, sdSlope, ptCut);
  }

  struct CountMiniDoubletConnections {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  ModulesConst modules,
                                  MiniDoublets mds,
                                  MiniDoubletsOccupancyConst mdsOccupancy,
                                  ObjectRangesConst ranges,
                                  const float ptCut) const {
      // The atomicAdd below with hierarchy::Threads{} requires one block in x, y dimensions.
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[1] == 1) &&
                        (alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[2] == 1));
      const auto& mdRanges = ranges.mdRanges();

      for (uint16_t innerLowerModuleIndex : cms::alpakatools::uniform_groups_z(acc, modules.nLowerModules())) {
        const unsigned int nInnerMDs = mdsOccupancy.nMDs()[innerLowerModuleIndex];
        if (nInnerMDs == 0)
          continue;

        const uint16_t nConnectedModules = modules.nConnectedModules()[innerLowerModuleIndex];
        if (nConnectedModules == 0)
          continue;

        for (uint16_t outerLowerModuleArrayIdx : cms::alpakatools::uniform_elements_y(acc, nConnectedModules)) {
          const uint16_t outerLowerModuleIndex = modules.moduleMap()[innerLowerModuleIndex][outerLowerModuleArrayIdx];
          const unsigned int nOuterMDs = mdsOccupancy.nMDs()[outerLowerModuleIndex];
          if (nOuterMDs == 0)
            continue;

          const unsigned int limit = nInnerMDs * nOuterMDs;

          for (unsigned int hitIndex : cms::alpakatools::uniform_elements_x(acc, limit)) {
            const unsigned int innerMDArrayIdx = hitIndex / nOuterMDs;
            const unsigned int outerMDArrayIdx = hitIndex % nOuterMDs;

            const unsigned int innerMDIndex = mdRanges[innerLowerModuleIndex][0] + innerMDArrayIdx;
            const unsigned int outerMDIndex = mdRanges[outerLowerModuleIndex][0] + outerMDArrayIdx;

            // Increment the connected max if the LS passes the delta phi cuts.
            if (passDeltaPhiCutsSelector(
                    acc, modules, mds, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex, ptCut)) {
              alpaka::atomicAdd(acc, &mds.connectedMax()[innerMDIndex], 1u, alpaka::hierarchy::Threads{});
            }
          }
        }
      }
    }
  };

  struct CreateSegmentArrayRanges {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  ObjectRanges ranges,
                                  MiniDoubletsConst mds,
                                  MiniDoubletsOccupancyConst mdsOccupancy) const {
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      int& nTotalSegments = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      if (cms::alpakatools::once_per_block(acc))
        nTotalSegments = 0;
      alpaka::syncBlockThreads(acc);

      for (uint16_t innerLowerModuleIndex : cms::alpakatools::uniform_elements(acc, modules.nLowerModules())) {
        if (modules.nConnectedModules()[innerLowerModuleIndex] == 0) {
          ranges.segmentModuleIndices()[innerLowerModuleIndex] = nTotalSegments;
          ranges.segmentModuleOccupancy()[innerLowerModuleIndex] = 0;
          continue;
        }

        // Sum the connected counts of all MDs in this module.
        const unsigned int nInnerMDs = mdsOccupancy.nMDs()[innerLowerModuleIndex];
        int occupancy = 0;
        if (nInnerMDs != 0) {
          const unsigned int firstMD = ranges.mdRanges()[innerLowerModuleIndex][0];
          for (unsigned int j = 0; j < nInnerMDs; ++j) {
            occupancy += mds.connectedMax()[firstMD + j];
          }
        }

        const int nTotSegs = alpaka::atomicAdd(acc, &nTotalSegments, occupancy, alpaka::hierarchy::Threads{});
        ranges.segmentModuleIndices()[innerLowerModuleIndex] = nTotSegs;
        ranges.segmentModuleOccupancy()[innerLowerModuleIndex] = occupancy;
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
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  SegmentsOccupancyConst segmentsOccupancy,
                                  ObjectRanges ranges) const {
      // implementation is 1D with a single block
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      for (uint16_t i : cms::alpakatools::uniform_elements(acc, modules.nLowerModules())) {
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
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  ObjectRangesConst ranges,
                                  HitsBaseConst hitsBase,
                                  HitsExtendedConst hitsExtended,
                                  PixelSeedsConst pixelSeeds,
                                  MiniDoublets mds,
                                  Segments segments,
                                  PixelSegments pixelSegments,
                                  uint16_t pixelModuleIndex,
                                  int size) const {
      for (int tid : cms::alpakatools::uniform_elements(acc, size)) {
        unsigned int innerMDIndex = ranges.miniDoubletModuleIndices()[pixelModuleIndex] + 2 * (tid);
        unsigned int outerMDIndex = ranges.miniDoubletModuleIndices()[pixelModuleIndex] + 2 * (tid) + 1;
        unsigned int pixelSegmentIndex = ranges.segmentModuleIndices()[pixelModuleIndex] + tid;

        addMDToMemory(acc,
                      mds,
                      hitsBase,
                      hitsExtended,
                      modules,
                      pixelSeeds.hitIndices()[tid][0],
                      pixelSeeds.hitIndices()[tid][1],
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
                      hitsBase,
                      hitsExtended,
                      modules,
                      pixelSeeds.hitIndices()[tid][2],
                      pixelSeeds.hitIndices()[tid][3],
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
        float slope = alpaka::math::sinh(acc, hitsBase.ys()[mds.outerHitIndices()[innerMDIndex]]);
        float intercept = hitsBase.zs()[mds.anchorHitIndices()[innerMDIndex]] -
                          slope * hitsExtended.rts()[mds.anchorHitIndices()[innerMDIndex]];
        float score_lsq = (hitsExtended.rts()[mds.anchorHitIndices()[outerMDIndex]] * slope + intercept) -
                          (hitsBase.zs()[mds.anchorHitIndices()[outerMDIndex]]);
        score_lsq = score_lsq * score_lsq;

        const Params_pLS::ArrayUxHits hits1{{hitsBase.idxs()[mds.anchorHitIndices()[innerMDIndex]],
                                             hitsBase.idxs()[mds.anchorHitIndices()[outerMDIndex]],
                                             hitsBase.idxs()[mds.outerHitIndices()[innerMDIndex]],
                                             hitsBase.idxs()[mds.outerHitIndices()[outerMDIndex]]}};
        addPixelSegmentToMemory(acc,
                                segments,
                                pixelSegments,
                                pixelSeeds,
                                mds,
                                innerMDIndex,
                                outerMDIndex,
                                pixelModuleIndex,
                                hits1,
                                pixelSeeds.hitIndices()[tid][0],
                                pixelSeeds.hitIndices()[tid][2],
                                pixelSeeds.deltaPhi()[tid],
                                pixelSegmentIndex,
                                tid,
                                score_lsq);
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif
