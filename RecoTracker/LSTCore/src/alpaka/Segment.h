#ifndef Segment_cuh
#define Segment_cuh

#ifdef LST_IS_CMSSW_PACKAGE
#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/alpaka/Module.h"
#else
#include "Constants.h"
#include "Module.h"
#endif

#include "EndcapGeometry.h"
#include "MiniDoublet.h"
#include "Hit.h"

namespace SDL {
  struct segments {
    FPX* dPhis;
    FPX* dPhiMins;
    FPX* dPhiMaxs;
    FPX* dPhiChanges;
    FPX* dPhiChangeMins;
    FPX* dPhiChangeMaxs;
    uint16_t* innerLowerModuleIndices;
    uint16_t* outerLowerModuleIndices;
    unsigned int* seedIdx;
    unsigned int* mdIndices;
    unsigned int* nMemoryLocations;
    unsigned int* innerMiniDoubletAnchorHitIndices;
    unsigned int* outerMiniDoubletAnchorHitIndices;
    int* charge;
    int* superbin;
    unsigned int* nSegments;             //number of segments per inner lower module
    unsigned int* totOccupancySegments;  //number of segments per inner lower module
    uint4* pLSHitsIdxs;
    int8_t* pixelType;
    char* isQuad;
    char* isDup;
    bool* partOfPT5;
    float* ptIn;
    float* ptErr;
    float* px;
    float* py;
    float* pz;
    float* etaErr;
    float* eta;
    float* phi;
    float* score;
    float* circleCenterX;
    float* circleCenterY;
    float* circleRadius;

    template <typename TBuff>
    void setData(TBuff& segmentsbuf) {
      dPhis = alpaka::getPtrNative(segmentsbuf.dPhis_buf);
      dPhiMins = alpaka::getPtrNative(segmentsbuf.dPhiMins_buf);
      dPhiMaxs = alpaka::getPtrNative(segmentsbuf.dPhiMaxs_buf);
      dPhiChanges = alpaka::getPtrNative(segmentsbuf.dPhiChanges_buf);
      dPhiChangeMins = alpaka::getPtrNative(segmentsbuf.dPhiChangeMins_buf);
      dPhiChangeMaxs = alpaka::getPtrNative(segmentsbuf.dPhiChangeMaxs_buf);
      innerLowerModuleIndices = alpaka::getPtrNative(segmentsbuf.innerLowerModuleIndices_buf);
      outerLowerModuleIndices = alpaka::getPtrNative(segmentsbuf.outerLowerModuleIndices_buf);
      seedIdx = alpaka::getPtrNative(segmentsbuf.seedIdx_buf);
      mdIndices = alpaka::getPtrNative(segmentsbuf.mdIndices_buf);
      nMemoryLocations = alpaka::getPtrNative(segmentsbuf.nMemoryLocations_buf);
      innerMiniDoubletAnchorHitIndices = alpaka::getPtrNative(segmentsbuf.innerMiniDoubletAnchorHitIndices_buf);
      outerMiniDoubletAnchorHitIndices = alpaka::getPtrNative(segmentsbuf.outerMiniDoubletAnchorHitIndices_buf);
      charge = alpaka::getPtrNative(segmentsbuf.charge_buf);
      superbin = alpaka::getPtrNative(segmentsbuf.superbin_buf);
      nSegments = alpaka::getPtrNative(segmentsbuf.nSegments_buf);
      totOccupancySegments = alpaka::getPtrNative(segmentsbuf.totOccupancySegments_buf);
      pLSHitsIdxs = alpaka::getPtrNative(segmentsbuf.pLSHitsIdxs_buf);
      pixelType = alpaka::getPtrNative(segmentsbuf.pixelType_buf);
      isQuad = alpaka::getPtrNative(segmentsbuf.isQuad_buf);
      isDup = alpaka::getPtrNative(segmentsbuf.isDup_buf);
      partOfPT5 = alpaka::getPtrNative(segmentsbuf.partOfPT5_buf);
      ptIn = alpaka::getPtrNative(segmentsbuf.ptIn_buf);
      ptErr = alpaka::getPtrNative(segmentsbuf.ptErr_buf);
      px = alpaka::getPtrNative(segmentsbuf.px_buf);
      py = alpaka::getPtrNative(segmentsbuf.py_buf);
      pz = alpaka::getPtrNative(segmentsbuf.pz_buf);
      etaErr = alpaka::getPtrNative(segmentsbuf.etaErr_buf);
      eta = alpaka::getPtrNative(segmentsbuf.eta_buf);
      phi = alpaka::getPtrNative(segmentsbuf.phi_buf);
      score = alpaka::getPtrNative(segmentsbuf.score_buf);
      circleCenterX = alpaka::getPtrNative(segmentsbuf.circleCenterX_buf);
      circleCenterY = alpaka::getPtrNative(segmentsbuf.circleCenterY_buf);
      circleRadius = alpaka::getPtrNative(segmentsbuf.circleRadius_buf);
    }
  };

  template <typename TDev>
  struct segmentsBuffer : segments {
    Buf<TDev, FPX> dPhis_buf;
    Buf<TDev, FPX> dPhiMins_buf;
    Buf<TDev, FPX> dPhiMaxs_buf;
    Buf<TDev, FPX> dPhiChanges_buf;
    Buf<TDev, FPX> dPhiChangeMins_buf;
    Buf<TDev, FPX> dPhiChangeMaxs_buf;
    Buf<TDev, uint16_t> innerLowerModuleIndices_buf;
    Buf<TDev, uint16_t> outerLowerModuleIndices_buf;
    Buf<TDev, unsigned int> seedIdx_buf;
    Buf<TDev, unsigned int> mdIndices_buf;
    Buf<TDev, unsigned int> nMemoryLocations_buf;
    Buf<TDev, unsigned int> innerMiniDoubletAnchorHitIndices_buf;
    Buf<TDev, unsigned int> outerMiniDoubletAnchorHitIndices_buf;
    Buf<TDev, int> charge_buf;
    Buf<TDev, int> superbin_buf;
    Buf<TDev, unsigned int> nSegments_buf;
    Buf<TDev, unsigned int> totOccupancySegments_buf;
    Buf<TDev, uint4> pLSHitsIdxs_buf;
    Buf<TDev, int8_t> pixelType_buf;
    Buf<TDev, char> isQuad_buf;
    Buf<TDev, char> isDup_buf;
    Buf<TDev, bool> partOfPT5_buf;
    Buf<TDev, float> ptIn_buf;
    Buf<TDev, float> ptErr_buf;
    Buf<TDev, float> px_buf;
    Buf<TDev, float> py_buf;
    Buf<TDev, float> pz_buf;
    Buf<TDev, float> etaErr_buf;
    Buf<TDev, float> eta_buf;
    Buf<TDev, float> phi_buf;
    Buf<TDev, float> score_buf;
    Buf<TDev, float> circleCenterX_buf;
    Buf<TDev, float> circleCenterY_buf;
    Buf<TDev, float> circleRadius_buf;

    template <typename TQueue, typename TDevAcc>
    segmentsBuffer(unsigned int nMemoryLocationsIn,
                   uint16_t nLowerModules,
                   unsigned int maxPixelSegments,
                   TDevAcc const& devAccIn,
                   TQueue& queue)
        : dPhis_buf(allocBufWrapper<FPX>(devAccIn, nMemoryLocationsIn, queue)),
          dPhiMins_buf(allocBufWrapper<FPX>(devAccIn, nMemoryLocationsIn, queue)),
          dPhiMaxs_buf(allocBufWrapper<FPX>(devAccIn, nMemoryLocationsIn, queue)),
          dPhiChanges_buf(allocBufWrapper<FPX>(devAccIn, nMemoryLocationsIn, queue)),
          dPhiChangeMins_buf(allocBufWrapper<FPX>(devAccIn, nMemoryLocationsIn, queue)),
          dPhiChangeMaxs_buf(allocBufWrapper<FPX>(devAccIn, nMemoryLocationsIn, queue)),
          innerLowerModuleIndices_buf(allocBufWrapper<uint16_t>(devAccIn, nMemoryLocationsIn, queue)),
          outerLowerModuleIndices_buf(allocBufWrapper<uint16_t>(devAccIn, nMemoryLocationsIn, queue)),
          seedIdx_buf(allocBufWrapper<unsigned int>(devAccIn, maxPixelSegments, queue)),
          mdIndices_buf(allocBufWrapper<unsigned int>(devAccIn, nMemoryLocationsIn * 2, queue)),
          nMemoryLocations_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          innerMiniDoubletAnchorHitIndices_buf(allocBufWrapper<unsigned int>(devAccIn, nMemoryLocationsIn, queue)),
          outerMiniDoubletAnchorHitIndices_buf(allocBufWrapper<unsigned int>(devAccIn, nMemoryLocationsIn, queue)),
          charge_buf(allocBufWrapper<int>(devAccIn, maxPixelSegments, queue)),
          superbin_buf(allocBufWrapper<int>(devAccIn, maxPixelSegments, queue)),
          nSegments_buf(allocBufWrapper<unsigned int>(devAccIn, nLowerModules + 1, queue)),
          totOccupancySegments_buf(allocBufWrapper<unsigned int>(devAccIn, nLowerModules + 1, queue)),
          pLSHitsIdxs_buf(allocBufWrapper<uint4>(devAccIn, maxPixelSegments, queue)),
          pixelType_buf(allocBufWrapper<int8_t>(devAccIn, maxPixelSegments, queue)),
          isQuad_buf(allocBufWrapper<char>(devAccIn, maxPixelSegments, queue)),
          isDup_buf(allocBufWrapper<char>(devAccIn, maxPixelSegments, queue)),
          partOfPT5_buf(allocBufWrapper<bool>(devAccIn, maxPixelSegments, queue)),
          ptIn_buf(allocBufWrapper<float>(devAccIn, maxPixelSegments, queue)),
          ptErr_buf(allocBufWrapper<float>(devAccIn, maxPixelSegments, queue)),
          px_buf(allocBufWrapper<float>(devAccIn, maxPixelSegments, queue)),
          py_buf(allocBufWrapper<float>(devAccIn, maxPixelSegments, queue)),
          pz_buf(allocBufWrapper<float>(devAccIn, maxPixelSegments, queue)),
          etaErr_buf(allocBufWrapper<float>(devAccIn, maxPixelSegments, queue)),
          eta_buf(allocBufWrapper<float>(devAccIn, maxPixelSegments, queue)),
          phi_buf(allocBufWrapper<float>(devAccIn, maxPixelSegments, queue)),
          score_buf(allocBufWrapper<float>(devAccIn, maxPixelSegments, queue)),
          circleCenterX_buf(allocBufWrapper<float>(devAccIn, maxPixelSegments, queue)),
          circleCenterY_buf(allocBufWrapper<float>(devAccIn, maxPixelSegments, queue)),
          circleRadius_buf(allocBufWrapper<float>(devAccIn, maxPixelSegments, queue)) {
      alpaka::memset(queue, nSegments_buf, 0u);
      alpaka::memset(queue, totOccupancySegments_buf, 0u);
      alpaka::memset(queue, partOfPT5_buf, false);
      alpaka::memset(queue, pLSHitsIdxs_buf, 0u);
      alpaka::wait(queue);
    }
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float isTighterTiltedModules_seg(struct SDL::modules& modulesInGPU,
                                                                  unsigned int moduleIndex) {
    // The "tighter" tilted modules are the subset of tilted modules that have smaller spacing
    // This is the same as what was previously considered as"isNormalTiltedModules"
    // See Figure 9.1 of https://cds.cern.ch/record/2272264/files/CMS-TDR-014.pdf
    short subdet = modulesInGPU.subdets[moduleIndex];
    short layer = modulesInGPU.layers[moduleIndex];
    short side = modulesInGPU.sides[moduleIndex];
    short rod = modulesInGPU.rods[moduleIndex];

    return (subdet == Barrel) && (((side != Center) && (layer == 3)) ||
                                  ((side == NegZ) && (((layer == 2) && (rod > 5)) || ((layer == 1) && (rod > 9)))) ||
                                  ((side == PosZ) && (((layer == 2) && (rod < 8)) || ((layer == 1) && (rod < 4)))));
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float isTighterTiltedModules_seg(short subdet, short layer, short side, short rod) {
    // The "tighter" tilted modules are the subset of tilted modules that have smaller spacing
    // This is the same as what was previously considered as"isNormalTiltedModules"
    // See Figure 9.1 of https://cds.cern.ch/record/2272264/files/CMS-TDR-014.pdf
    return (subdet == Barrel) && (((side != Center) && (layer == 3)) ||
                                  ((side == NegZ) && (((layer == 2) && (rod > 5)) || ((layer == 1) && (rod > 9)))) ||
                                  ((side == PosZ) && (((layer == 2) && (rod < 8)) || ((layer == 1) && (rod < 4)))));
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float moduleGapSize_seg(short layer, short ring, short subdet, short side, short rod) {
    static constexpr float miniDeltaTilted[3] = {0.26f, 0.26f, 0.26f};
    static constexpr float miniDeltaFlat[6] = {0.26f, 0.16f, 0.16f, 0.18f, 0.18f, 0.18f};
    static constexpr float miniDeltaLooseTilted[3] = {0.4f, 0.4f, 0.4f};
    static constexpr float miniDeltaEndcap[5][15] = {
        {0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, /*10*/ 0.18f, 0.18f, 0.18f, 0.18f, 0.18f},
        {0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, /*10*/ 0.18f, 0.18f, 0.18f, 0.18f, 0.18f},
        {0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.18f, 0.18f, /*10*/ 0.18f, 0.18f, 0.18f, 0.18f, 0.18f},
        {0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.18f, 0.18f, /*10*/ 0.18f, 0.18f, 0.18f, 0.18f, 0.18f},
        {0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.18f, /*10*/ 0.18f, 0.18f, 0.18f, 0.18f, 0.18f}};

    unsigned int iL = layer - 1;
    unsigned int iR = ring - 1;

    float moduleSeparation = 0;

    if (subdet == Barrel and side == Center) {
      moduleSeparation = miniDeltaFlat[iL];
    } else if (isTighterTiltedModules_seg(subdet, layer, side, rod)) {
      moduleSeparation = miniDeltaTilted[iL];
    } else if (subdet == Endcap) {
      moduleSeparation = miniDeltaEndcap[iL][iR];
    } else  //Loose tilted modules
    {
      moduleSeparation = miniDeltaLooseTilted[iL];
    }

    return moduleSeparation;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float moduleGapSize_seg(struct SDL::modules& modulesInGPU, unsigned int moduleIndex) {
    static constexpr float miniDeltaTilted[3] = {0.26f, 0.26f, 0.26f};
    static constexpr float miniDeltaFlat[6] = {0.26f, 0.16f, 0.16f, 0.18f, 0.18f, 0.18f};
    static constexpr float miniDeltaLooseTilted[3] = {0.4f, 0.4f, 0.4f};
    static constexpr float miniDeltaEndcap[5][15] = {
        {0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, /*10*/ 0.18f, 0.18f, 0.18f, 0.18f, 0.18f},
        {0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, /*10*/ 0.18f, 0.18f, 0.18f, 0.18f, 0.18f},
        {0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.18f, 0.18f, /*10*/ 0.18f, 0.18f, 0.18f, 0.18f, 0.18f},
        {0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.18f, 0.18f, /*10*/ 0.18f, 0.18f, 0.18f, 0.18f, 0.18f},
        {0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.18f, /*10*/ 0.18f, 0.18f, 0.18f, 0.18f, 0.18f}};

    unsigned int iL = modulesInGPU.layers[moduleIndex] - 1;
    unsigned int iR = modulesInGPU.rings[moduleIndex] - 1;
    short subdet = modulesInGPU.subdets[moduleIndex];
    short side = modulesInGPU.sides[moduleIndex];

    float moduleSeparation = 0;

    if (subdet == Barrel and side == Center) {
      moduleSeparation = miniDeltaFlat[iL];
    } else if (isTighterTiltedModules_seg(modulesInGPU, moduleIndex)) {
      moduleSeparation = miniDeltaTilted[iL];
    } else if (subdet == Endcap) {
      moduleSeparation = miniDeltaEndcap[iL][iR];
    } else  //Loose tilted modules
    {
      moduleSeparation = miniDeltaLooseTilted[iL];
    }

    return moduleSeparation;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void dAlphaThreshold(TAcc const& acc,
                                                      float* dAlphaThresholdValues,
                                                      struct SDL::modules& modulesInGPU,
                                                      struct SDL::miniDoublets& mdsInGPU,
                                                      float& xIn,
                                                      float& yIn,
                                                      float& zIn,
                                                      float& rtIn,
                                                      float& xOut,
                                                      float& yOut,
                                                      float& zOut,
                                                      float& rtOut,
                                                      uint16_t& innerLowerModuleIndex,
                                                      uint16_t& outerLowerModuleIndex,
                                                      unsigned int& innerMDIndex,
                                                      unsigned int& outerMDIndex) {
    float sdMuls = (modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel)
                       ? miniMulsPtScaleBarrel[modulesInGPU.layers[innerLowerModuleIndex] - 1] * 3.f / ptCut
                       : miniMulsPtScaleEndcap[modulesInGPU.layers[innerLowerModuleIndex] - 1] * 3.f / ptCut;

    //more accurate then outer rt - inner rt
    float segmentDr = alpaka::math::sqrt(acc, (yOut - yIn) * (yOut - yIn) + (xOut - xIn) * (xOut - xIn));

    const float dAlpha_Bfield =
        alpaka::math::asin(acc, alpaka::math::min(acc, segmentDr * k2Rinv1GeVf / ptCut, sinAlphaMax));

    bool isInnerTilted = modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel and
                         modulesInGPU.sides[innerLowerModuleIndex] != SDL::Center;
    bool isOuterTilted = modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Barrel and
                         modulesInGPU.sides[outerLowerModuleIndex] != SDL::Center;

    const float& drdzInner = modulesInGPU.drdzs[innerLowerModuleIndex];
    const float& drdzOuter = modulesInGPU.drdzs[outerLowerModuleIndex];
    float innerModuleGapSize = SDL::moduleGapSize_seg(modulesInGPU, innerLowerModuleIndex);
    float outerModuleGapSize = SDL::moduleGapSize_seg(modulesInGPU, outerLowerModuleIndex);
    const float innerminiTilt = isInnerTilted
                                    ? (0.5f * pixelPSZpitch * drdzInner /
                                       alpaka::math::sqrt(acc, 1.f + drdzInner * drdzInner) / innerModuleGapSize)
                                    : 0;

    const float outerminiTilt = isOuterTilted
                                    ? (0.5f * pixelPSZpitch * drdzOuter /
                                       alpaka::math::sqrt(acc, 1.f + drdzOuter * drdzOuter) / outerModuleGapSize)
                                    : 0;

    float miniDelta = innerModuleGapSize;

    float sdLumForInnerMini;
    float sdLumForOuterMini;

    if (modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel) {
      sdLumForInnerMini = innerminiTilt * dAlpha_Bfield;
    } else {
      sdLumForInnerMini = mdsInGPU.dphis[innerMDIndex] * 15.0f / mdsInGPU.dzs[innerMDIndex];
    }

    if (modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Barrel) {
      sdLumForOuterMini = outerminiTilt * dAlpha_Bfield;
    } else {
      sdLumForOuterMini = mdsInGPU.dphis[outerMDIndex] * 15.0f / mdsInGPU.dzs[outerMDIndex];
    }

    // Unique stuff for the segment dudes alone
    float dAlpha_res_inner =
        0.02f / miniDelta *
        (modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel ? 1.0f : alpaka::math::abs(acc, zIn) / rtIn);
    float dAlpha_res_outer =
        0.02f / miniDelta *
        (modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Barrel ? 1.0f : alpaka::math::abs(acc, zOut) / rtOut);

    float dAlpha_res = dAlpha_res_inner + dAlpha_res_outer;

    if (modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel and
        modulesInGPU.sides[innerLowerModuleIndex] == SDL::Center) {
      dAlphaThresholdValues[0] = dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls);
    } else {
      dAlphaThresholdValues[0] =
          dAlpha_Bfield +
          alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls + sdLumForInnerMini * sdLumForInnerMini);
    }

    if (modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Barrel and
        modulesInGPU.sides[outerLowerModuleIndex] == SDL::Center) {
      dAlphaThresholdValues[1] = dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls);
    } else {
      dAlphaThresholdValues[1] =
          dAlpha_Bfield +
          alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls + sdLumForOuterMini * sdLumForOuterMini);
    }

    //Inner to outer
    dAlphaThresholdValues[2] = dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls);
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addSegmentToMemory(struct SDL::segments& segmentsInGPU,
                                                         unsigned int lowerMDIndex,
                                                         unsigned int upperMDIndex,
                                                         uint16_t innerLowerModuleIndex,
                                                         uint16_t outerLowerModuleIndex,
                                                         unsigned int innerMDAnchorHitIndex,
                                                         unsigned int outerMDAnchorHitIndex,
                                                         float& dPhi,
                                                         float& dPhiMin,
                                                         float& dPhiMax,
                                                         float& dPhiChange,
                                                         float& dPhiChangeMin,
                                                         float& dPhiChangeMax,
                                                         unsigned int idx) {
    //idx will be computed in the kernel, which is the index into which the
    //segment will be written
    //nSegments will be incremented in the kernel
    //printf("seg: %u %u %u %u\n",lowerMDIndex, upperMDIndex,innerLowerModuleIndex,outerLowerModuleIndex);
    segmentsInGPU.mdIndices[idx * 2] = lowerMDIndex;
    segmentsInGPU.mdIndices[idx * 2 + 1] = upperMDIndex;
    segmentsInGPU.innerLowerModuleIndices[idx] = innerLowerModuleIndex;
    segmentsInGPU.outerLowerModuleIndices[idx] = outerLowerModuleIndex;
    segmentsInGPU.innerMiniDoubletAnchorHitIndices[idx] = innerMDAnchorHitIndex;
    segmentsInGPU.outerMiniDoubletAnchorHitIndices[idx] = outerMDAnchorHitIndex;

    segmentsInGPU.dPhis[idx] = __F2H(dPhi);
    segmentsInGPU.dPhiMins[idx] = __F2H(dPhiMin);
    segmentsInGPU.dPhiMaxs[idx] = __F2H(dPhiMax);
    segmentsInGPU.dPhiChanges[idx] = __F2H(dPhiChange);
    segmentsInGPU.dPhiChangeMins[idx] = __F2H(dPhiChangeMin);
    segmentsInGPU.dPhiChangeMaxs[idx] = __F2H(dPhiChangeMax);
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addPixelSegmentToMemory(TAcc const& acc,
                                                              struct SDL::segments& segmentsInGPU,
                                                              struct SDL::miniDoublets& mdsInGPU,
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
    segmentsInGPU.mdIndices[idx * 2] = innerMDIndex;
    segmentsInGPU.mdIndices[idx * 2 + 1] = outerMDIndex;
    segmentsInGPU.innerLowerModuleIndices[idx] = pixelModuleIndex;
    segmentsInGPU.outerLowerModuleIndices[idx] = pixelModuleIndex;
    segmentsInGPU.innerMiniDoubletAnchorHitIndices[idx] = innerAnchorHitIndex;
    segmentsInGPU.outerMiniDoubletAnchorHitIndices[idx] = outerAnchorHitIndex;
    segmentsInGPU.dPhiChanges[idx] = __F2H(dPhiChange);
    segmentsInGPU.isDup[pixelSegmentArrayIndex] = false;
    segmentsInGPU.score[pixelSegmentArrayIndex] = score;

    segmentsInGPU.pLSHitsIdxs[pixelSegmentArrayIndex].x = hitIdxs[0];
    segmentsInGPU.pLSHitsIdxs[pixelSegmentArrayIndex].y = hitIdxs[1];
    segmentsInGPU.pLSHitsIdxs[pixelSegmentArrayIndex].z = hitIdxs[2];
    segmentsInGPU.pLSHitsIdxs[pixelSegmentArrayIndex].w = hitIdxs[3];

    //computing circle parameters
    /*
        The two anchor hits are r3PCA and r3LH. p3PCA pt, eta, phi is hitIndex1 x, y, z
        */
    float circleRadius = mdsInGPU.outerX[innerMDIndex] / (2 * k2Rinv1GeVf);
    float circlePhi = mdsInGPU.outerZ[innerMDIndex];
    float candidateCenterXs[] = {mdsInGPU.anchorX[innerMDIndex] + circleRadius * alpaka::math::sin(acc, circlePhi),
                                 mdsInGPU.anchorX[innerMDIndex] - circleRadius * alpaka::math::sin(acc, circlePhi)};
    float candidateCenterYs[] = {mdsInGPU.anchorY[innerMDIndex] - circleRadius * alpaka::math::cos(acc, circlePhi),
                                 mdsInGPU.anchorY[innerMDIndex] + circleRadius * alpaka::math::cos(acc, circlePhi)};

    //check which of the circles can accommodate r3LH better (we won't get perfect agreement)
    float bestChiSquared = SDL::SDL_INF;
    float chiSquared;
    size_t bestIndex;
    for (size_t i = 0; i < 2; i++) {
      chiSquared =
          alpaka::math::abs(acc,
                            alpaka::math::sqrt(acc,
                                               (mdsInGPU.anchorX[outerMDIndex] - candidateCenterXs[i]) *
                                                       (mdsInGPU.anchorX[outerMDIndex] - candidateCenterXs[i]) +
                                                   (mdsInGPU.anchorY[outerMDIndex] - candidateCenterYs[i]) *
                                                       (mdsInGPU.anchorY[outerMDIndex] - candidateCenterYs[i])) -
                                circleRadius);
      if (chiSquared < bestChiSquared) {
        bestChiSquared = chiSquared;
        bestIndex = i;
      }
    }
    segmentsInGPU.circleCenterX[pixelSegmentArrayIndex] = candidateCenterXs[bestIndex];
    segmentsInGPU.circleCenterY[pixelSegmentArrayIndex] = candidateCenterYs[bestIndex];
    segmentsInGPU.circleRadius[pixelSegmentArrayIndex] = circleRadius;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runSegmentDefaultAlgoBarrel(TAcc const& acc,
                                                                  struct SDL::modules& modulesInGPU,
                                                                  struct SDL::miniDoublets& mdsInGPU,
                                                                  uint16_t& innerLowerModuleIndex,
                                                                  uint16_t& outerLowerModuleIndex,
                                                                  unsigned int& innerMDIndex,
                                                                  unsigned int& outerMDIndex,
                                                                  float& zIn,
                                                                  float& zOut,
                                                                  float& rtIn,
                                                                  float& rtOut,
                                                                  float& dPhi,
                                                                  float& dPhiMin,
                                                                  float& dPhiMax,
                                                                  float& dPhiChange,
                                                                  float& dPhiChangeMin,
                                                                  float& dPhiChangeMax,
                                                                  float& dAlphaInnerMDSegment,
                                                                  float& dAlphaOuterMDSegment,
                                                                  float& dAlphaInnerMDOuterMD,
                                                                  float& zLo,
                                                                  float& zHi,
                                                                  float& sdCut,
                                                                  float& dAlphaInnerMDSegmentThreshold,
                                                                  float& dAlphaOuterMDSegmentThreshold,
                                                                  float& dAlphaInnerMDOuterMDThreshold) {
    bool pass = true;

    float sdMuls = (modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel)
                       ? miniMulsPtScaleBarrel[modulesInGPU.layers[innerLowerModuleIndex] - 1] * 3.f / ptCut
                       : miniMulsPtScaleEndcap[modulesInGPU.layers[innerLowerModuleIndex] - 1] * 3.f / ptCut;

    float xIn, yIn, xOut, yOut;

    xIn = mdsInGPU.anchorX[innerMDIndex];
    yIn = mdsInGPU.anchorY[innerMDIndex];
    zIn = mdsInGPU.anchorZ[innerMDIndex];
    rtIn = mdsInGPU.anchorRt[innerMDIndex];

    xOut = mdsInGPU.anchorX[outerMDIndex];
    yOut = mdsInGPU.anchorY[outerMDIndex];
    zOut = mdsInGPU.anchorZ[outerMDIndex];
    rtOut = mdsInGPU.anchorRt[outerMDIndex];

    float sdSlope = alpaka::math::asin(acc, alpaka::math::min(acc, rtOut * k2Rinv1GeVf / ptCut, sinAlphaMax));
    float sdPVoff = 0.1f / rtOut;
    float dzDrtScale = alpaka::math::tan(acc, sdSlope) / sdSlope;  //FIXME: need appropriate value

    const float zGeom = modulesInGPU.layers[innerLowerModuleIndex] <= 2 ? 2.f * pixelPSZpitch : 2.f * strip2SZpitch;

    zLo = zIn + (zIn - deltaZLum) * (rtOut / rtIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) -
          zGeom;  //slope-correction only on outer end
    zHi = zIn + (zIn + deltaZLum) * (rtOut / rtIn - 1.f) * (zIn < 0.f ? 1.f : dzDrtScale) + zGeom;

    pass = pass and ((zOut >= zLo) && (zOut <= zHi));
    if (not pass)
      return pass;

    sdCut = sdSlope + alpaka::math::sqrt(acc, sdMuls * sdMuls + sdPVoff * sdPVoff);

    dPhi = SDL::phi_mpi_pi(acc, mdsInGPU.anchorPhi[outerMDIndex] - mdsInGPU.anchorPhi[innerMDIndex]);

    pass = pass and (alpaka::math::abs(acc, dPhi) <= sdCut);
    if (not pass)
      return pass;

    dPhiChange = SDL::phi_mpi_pi(acc, SDL::phi(acc, xOut - xIn, yOut - yIn) - mdsInGPU.anchorPhi[innerMDIndex]);

    pass = pass and (alpaka::math::abs(acc, dPhiChange) <= sdCut);
    if (not pass)
      return pass;

    float dAlphaThresholdValues[3];
    dAlphaThreshold(acc,
                    dAlphaThresholdValues,
                    modulesInGPU,
                    mdsInGPU,
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

    float innerMDAlpha = mdsInGPU.dphichanges[innerMDIndex];
    float outerMDAlpha = mdsInGPU.dphichanges[outerMDIndex];
    dAlphaInnerMDSegment = innerMDAlpha - dPhiChange;
    dAlphaOuterMDSegment = outerMDAlpha - dPhiChange;
    dAlphaInnerMDOuterMD = innerMDAlpha - outerMDAlpha;

    dAlphaInnerMDSegmentThreshold = dAlphaThresholdValues[0];
    dAlphaOuterMDSegmentThreshold = dAlphaThresholdValues[1];
    dAlphaInnerMDOuterMDThreshold = dAlphaThresholdValues[2];

    pass = pass and (alpaka::math::abs(acc, dAlphaInnerMDSegment) < dAlphaInnerMDSegmentThreshold);
    if (not pass)
      return pass;
    pass = pass and (alpaka::math::abs(acc, dAlphaOuterMDSegment) < dAlphaOuterMDSegmentThreshold);
    if (not pass)
      return pass;
    pass = pass and (alpaka::math::abs(acc, dAlphaInnerMDOuterMD) < dAlphaInnerMDOuterMDThreshold);

    return pass;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runSegmentDefaultAlgoEndcap(TAcc const& acc,
                                                                  struct SDL::modules& modulesInGPU,
                                                                  struct SDL::miniDoublets& mdsInGPU,
                                                                  uint16_t& innerLowerModuleIndex,
                                                                  uint16_t& outerLowerModuleIndex,
                                                                  unsigned int& innerMDIndex,
                                                                  unsigned int& outerMDIndex,
                                                                  float& zIn,
                                                                  float& zOut,
                                                                  float& rtIn,
                                                                  float& rtOut,
                                                                  float& dPhi,
                                                                  float& dPhiMin,
                                                                  float& dPhiMax,
                                                                  float& dPhiChange,
                                                                  float& dPhiChangeMin,
                                                                  float& dPhiChangeMax,
                                                                  float& dAlphaInnerMDSegment,
                                                                  float& dAlphaOuterMDSegment,
                                                                  float& rtLo,
                                                                  float& rtHi,
                                                                  float& sdCut,
                                                                  float& dAlphaInnerMDSegmentThreshold,
                                                                  float& dAlphaOuterMDSegmentThreshold,
                                                                  float& dAlphaInnerMDOuterMDThreshold,
                                                                  float& dAlphaInnerMDOuterMD) {
    bool pass = true;

    float xIn, yIn;
    float xOut, yOut;

    xIn = mdsInGPU.anchorX[innerMDIndex];
    yIn = mdsInGPU.anchorY[innerMDIndex];
    zIn = mdsInGPU.anchorZ[innerMDIndex];
    rtIn = mdsInGPU.anchorRt[innerMDIndex];

    xOut = mdsInGPU.anchorX[outerMDIndex];
    yOut = mdsInGPU.anchorY[outerMDIndex];
    zOut = mdsInGPU.anchorZ[outerMDIndex];
    rtOut = mdsInGPU.anchorRt[outerMDIndex];

    bool outerLayerEndcapTwoS = (modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Endcap) &&
                                (modulesInGPU.moduleType[outerLowerModuleIndex] == SDL::TwoS);

    float sdSlope = alpaka::math::asin(acc, alpaka::math::min(acc, rtOut * k2Rinv1GeVf / ptCut, sinAlphaMax));
    float disks2SMinRadius = 60.f;

    float rtGeom = ((rtIn < disks2SMinRadius && rtOut < disks2SMinRadius)
                        ? (2.f * pixelPSZpitch)
                        : ((rtIn < disks2SMinRadius || rtOut < disks2SMinRadius) ? (pixelPSZpitch + strip2SZpitch)
                                                                                 : (2.f * strip2SZpitch)));

    //cut 0 - z compatibility
    pass = pass and (zIn * zOut >= 0);
    if (not pass)
      return pass;

    float dz = zOut - zIn;
    // Alpaka: Needs to be moved over
    float dLum = SDL::copysignf(deltaZLum, zIn);
    float drtDzScale = sdSlope / alpaka::math::tan(acc, sdSlope);

    rtLo = alpaka::math::max(
        acc, rtIn * (1.f + dz / (zIn + dLum) * drtDzScale) - rtGeom, rtIn - 0.5f * rtGeom);  //rt should increase
    rtHi = rtIn * (zOut - dLum) / (zIn - dLum) +
           rtGeom;  //dLum for luminous; rGeom for measurement size; no tanTheta_loc(pt) correction

    // Completeness
    pass = pass and ((rtOut >= rtLo) && (rtOut <= rtHi));
    if (not pass)
      return pass;

    dPhi = SDL::phi_mpi_pi(acc, mdsInGPU.anchorPhi[outerMDIndex] - mdsInGPU.anchorPhi[innerMDIndex]);

    sdCut = sdSlope;
    if (outerLayerEndcapTwoS) {
      float dPhiPos_high =
          SDL::phi_mpi_pi(acc, mdsInGPU.anchorHighEdgePhi[outerMDIndex] - mdsInGPU.anchorPhi[innerMDIndex]);
      float dPhiPos_low =
          SDL::phi_mpi_pi(acc, mdsInGPU.anchorLowEdgePhi[outerMDIndex] - mdsInGPU.anchorPhi[innerMDIndex]);

      dPhiMax = alpaka::math::abs(acc, dPhiPos_high) > alpaka::math::abs(acc, dPhiPos_low) ? dPhiPos_high : dPhiPos_low;
      dPhiMin = alpaka::math::abs(acc, dPhiPos_high) > alpaka::math::abs(acc, dPhiPos_low) ? dPhiPos_low : dPhiPos_high;
    } else {
      dPhiMax = dPhi;
      dPhiMin = dPhi;
    }
    pass = pass and (alpaka::math::abs(acc, dPhi) <= sdCut);
    if (not pass)
      return pass;

    float dzFrac = dz / zIn;
    dPhiChange = dPhi / dzFrac * (1.f + dzFrac);
    dPhiChangeMin = dPhiMin / dzFrac * (1.f + dzFrac);
    dPhiChangeMax = dPhiMax / dzFrac * (1.f + dzFrac);

    pass = pass and (alpaka::math::abs(acc, dPhiChange) <= sdCut);
    if (not pass)
      return pass;

    float dAlphaThresholdValues[3];
    dAlphaThreshold(acc,
                    dAlphaThresholdValues,
                    modulesInGPU,
                    mdsInGPU,
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

    dAlphaInnerMDSegmentThreshold = dAlphaThresholdValues[0];
    dAlphaOuterMDSegmentThreshold = dAlphaThresholdValues[1];
    dAlphaInnerMDOuterMDThreshold = dAlphaThresholdValues[2];

    float innerMDAlpha = mdsInGPU.dphichanges[innerMDIndex];
    float outerMDAlpha = mdsInGPU.dphichanges[outerMDIndex];
    dAlphaInnerMDSegment = innerMDAlpha - dPhiChange;
    dAlphaOuterMDSegment = outerMDAlpha - dPhiChange;
    dAlphaInnerMDOuterMD = innerMDAlpha - outerMDAlpha;

    pass = pass and (alpaka::math::abs(acc, dAlphaInnerMDSegment) < dAlphaThresholdValues[0]);
    if (not pass)
      return pass;
    pass = pass and (alpaka::math::abs(acc, dAlphaOuterMDSegment) < dAlphaThresholdValues[1]);
    if (not pass)
      return pass;
    pass = pass and (alpaka::math::abs(acc, dAlphaInnerMDOuterMD) < dAlphaThresholdValues[2]);

    return pass;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runSegmentDefaultAlgo(TAcc const& acc,
                                                            struct SDL::modules& modulesInGPU,
                                                            struct SDL::miniDoublets& mdsInGPU,
                                                            uint16_t& innerLowerModuleIndex,
                                                            uint16_t& outerLowerModuleIndex,
                                                            unsigned int& innerMDIndex,
                                                            unsigned int& outerMDIndex,
                                                            float& zIn,
                                                            float& zOut,
                                                            float& rtIn,
                                                            float& rtOut,
                                                            float& dPhi,
                                                            float& dPhiMin,
                                                            float& dPhiMax,
                                                            float& dPhiChange,
                                                            float& dPhiChangeMin,
                                                            float& dPhiChangeMax,
                                                            float& dAlphaInnerMDSegment,
                                                            float& dAlphaOuterMDSegment,
                                                            float& dAlphaInnerMDOuterMD,
                                                            float& zLo,
                                                            float& zHi,
                                                            float& rtLo,
                                                            float& rtHi,
                                                            float& sdCut,
                                                            float& dAlphaInnerMDSegmentThreshold,
                                                            float& dAlphaOuterMDSegmentThreshold,
                                                            float& dAlphaInnerMDOuterMDThreshold) {
    zLo = -999.f;
    zHi = -999.f;
    rtLo = -999.f;
    rtHi = -999.f;

    if (modulesInGPU.subdets[innerLowerModuleIndex] == SDL::Barrel and
        modulesInGPU.subdets[outerLowerModuleIndex] == SDL::Barrel) {
      return runSegmentDefaultAlgoBarrel(acc,
                                         modulesInGPU,
                                         mdsInGPU,
                                         innerLowerModuleIndex,
                                         outerLowerModuleIndex,
                                         innerMDIndex,
                                         outerMDIndex,
                                         zIn,
                                         zOut,
                                         rtIn,
                                         rtOut,
                                         dPhi,
                                         dPhiMin,
                                         dPhiMax,
                                         dPhiChange,
                                         dPhiChangeMin,
                                         dPhiChangeMax,
                                         dAlphaInnerMDSegment,
                                         dAlphaOuterMDSegment,
                                         dAlphaInnerMDOuterMD,
                                         zLo,
                                         zHi,
                                         sdCut,
                                         dAlphaInnerMDSegmentThreshold,
                                         dAlphaOuterMDSegmentThreshold,
                                         dAlphaInnerMDOuterMDThreshold);
    } else {
      return runSegmentDefaultAlgoEndcap(acc,
                                         modulesInGPU,
                                         mdsInGPU,
                                         innerLowerModuleIndex,
                                         outerLowerModuleIndex,
                                         innerMDIndex,
                                         outerMDIndex,
                                         zIn,
                                         zOut,
                                         rtIn,
                                         rtOut,
                                         dPhi,
                                         dPhiMin,
                                         dPhiMax,
                                         dPhiChange,
                                         dPhiChangeMin,
                                         dPhiChangeMax,
                                         dAlphaInnerMDSegment,
                                         dAlphaOuterMDSegment,
                                         dAlphaInnerMDOuterMD,
                                         rtLo,
                                         rtHi,
                                         sdCut,
                                         dAlphaInnerMDSegmentThreshold,
                                         dAlphaOuterMDSegmentThreshold,
                                         dAlphaInnerMDOuterMDThreshold);
    }
  };

  struct createSegmentsInGPUv2 {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  struct SDL::modules modulesInGPU,
                                  struct SDL::miniDoublets mdsInGPU,
                                  struct SDL::segments segmentsInGPU,
                                  struct SDL::objectRanges rangesInGPU) const {
      auto const globalBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
      auto const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
      auto const gridBlockExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc);
      auto const blockThreadExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);

      for (uint16_t innerLowerModuleIndex = globalBlockIdx[2]; innerLowerModuleIndex < (*modulesInGPU.nLowerModules);
           innerLowerModuleIndex += gridBlockExtent[2]) {
        unsigned int nInnerMDs = mdsInGPU.nMDs[innerLowerModuleIndex];
        if (nInnerMDs == 0)
          continue;

        unsigned int nConnectedModules = modulesInGPU.nConnectedModules[innerLowerModuleIndex];

        for (uint16_t outerLowerModuleArrayIdx = blockThreadIdx[1]; outerLowerModuleArrayIdx < nConnectedModules;
             outerLowerModuleArrayIdx += blockThreadExtent[1]) {
          uint16_t outerLowerModuleIndex =
              modulesInGPU.moduleMap[innerLowerModuleIndex * MAX_CONNECTED_MODULES + outerLowerModuleArrayIdx];

          unsigned int nOuterMDs = mdsInGPU.nMDs[outerLowerModuleIndex];

          unsigned int limit = nInnerMDs * nOuterMDs;

          if (limit == 0)
            continue;
          for (unsigned int hitIndex = blockThreadIdx[2]; hitIndex < limit; hitIndex += blockThreadExtent[2]) {
            unsigned int innerMDArrayIdx = hitIndex / nOuterMDs;
            unsigned int outerMDArrayIdx = hitIndex % nOuterMDs;
            if (outerMDArrayIdx >= nOuterMDs)
              continue;

            unsigned int innerMDIndex = rangesInGPU.mdRanges[innerLowerModuleIndex * 2] + innerMDArrayIdx;
            unsigned int outerMDIndex = rangesInGPU.mdRanges[outerLowerModuleIndex * 2] + outerMDArrayIdx;

            float zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax,
                dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD;

            unsigned int innerMiniDoubletAnchorHitIndex = mdsInGPU.anchorHitIndices[innerMDIndex];
            unsigned int outerMiniDoubletAnchorHitIndex = mdsInGPU.anchorHitIndices[outerMDIndex];
            dPhiMin = 0;
            dPhiMax = 0;
            dPhiChangeMin = 0;
            dPhiChangeMax = 0;
            float zLo, zHi, rtLo, rtHi, sdCut, dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold,
                dAlphaInnerMDOuterMDThreshold;
            bool pass = runSegmentDefaultAlgo(acc,
                                              modulesInGPU,
                                              mdsInGPU,
                                              innerLowerModuleIndex,
                                              outerLowerModuleIndex,
                                              innerMDIndex,
                                              outerMDIndex,
                                              zIn,
                                              zOut,
                                              rtIn,
                                              rtOut,
                                              dPhi,
                                              dPhiMin,
                                              dPhiMax,
                                              dPhiChange,
                                              dPhiChangeMin,
                                              dPhiChangeMax,
                                              dAlphaInnerMDSegment,
                                              dAlphaOuterMDSegment,
                                              dAlphaInnerMDOuterMD,
                                              zLo,
                                              zHi,
                                              rtLo,
                                              rtHi,
                                              sdCut,
                                              dAlphaInnerMDSegmentThreshold,
                                              dAlphaOuterMDSegmentThreshold,
                                              dAlphaInnerMDOuterMDThreshold);

            if (pass) {
              unsigned int totOccupancySegments = alpaka::atomicOp<alpaka::AtomicAdd>(
                  acc, &segmentsInGPU.totOccupancySegments[innerLowerModuleIndex], 1u);
              if (static_cast<int>(totOccupancySegments) >= rangesInGPU.segmentModuleOccupancy[innerLowerModuleIndex]) {
#ifdef Warnings
                printf("Segment excess alert! Module index = %d\n", innerLowerModuleIndex);
#endif
              } else {
                unsigned int segmentModuleIdx =
                    alpaka::atomicOp<alpaka::AtomicAdd>(acc, &segmentsInGPU.nSegments[innerLowerModuleIndex], 1u);
                unsigned int segmentIdx = rangesInGPU.segmentModuleIndices[innerLowerModuleIndex] + segmentModuleIdx;

                addSegmentToMemory(segmentsInGPU,
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

  struct createSegmentArrayRanges {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  struct SDL::modules modulesInGPU,
                                  struct SDL::objectRanges rangesInGPU,
                                  struct SDL::miniDoublets mdsInGPU) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      // Initialize variables in shared memory and set to 0
      int& nTotalSegments = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      nTotalSegments = 0;
      alpaka::syncBlockThreads(acc);

      // Initialize variables outside of the for loop.
      int occupancy, category_number, eta_number;

      for (uint16_t i = globalThreadIdx[2]; i < *modulesInGPU.nLowerModules; i += gridThreadExtent[2]) {
        if (modulesInGPU.nConnectedModules[i] == 0) {
          rangesInGPU.segmentModuleIndices[i] = nTotalSegments;
          rangesInGPU.segmentModuleOccupancy[i] = 0;
          continue;
        }

        short module_rings = modulesInGPU.rings[i];
        short module_layers = modulesInGPU.layers[i];
        short module_subdets = modulesInGPU.subdets[i];
        float module_eta = alpaka::math::abs(acc, modulesInGPU.eta[i]);

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

        if (module_eta < 0.75)
          eta_number = 0;
        else if (module_eta > 0.75 && module_eta < 1.5)
          eta_number = 1;
        else if (module_eta > 1.5 && module_eta < 2.25)
          eta_number = 2;
        else if (module_eta > 2.25 && module_eta < 3)
          eta_number = 3;
        else
          eta_number = -1;

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
#ifdef Warnings
          printf("Unhandled case in createSegmentArrayRanges! Module index = %i\n", i);
#endif
        }

        int nTotSegs = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &nTotalSegments, occupancy);
        rangesInGPU.segmentModuleIndices[i] = nTotSegs;
        rangesInGPU.segmentModuleOccupancy[i] = occupancy;
      }

      // Wait for all threads to finish before reporting final values
      alpaka::syncBlockThreads(acc);
      if (globalThreadIdx[2] == 0) {
        rangesInGPU.segmentModuleIndices[*modulesInGPU.nLowerModules] = nTotalSegments;
        *rangesInGPU.device_nTotalSegs = nTotalSegments;
      }
    }
  };

  struct addSegmentRangesToEventExplicit {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  struct SDL::modules modulesInGPU,
                                  struct SDL::segments segmentsInGPU,
                                  struct SDL::objectRanges rangesInGPU) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (uint16_t i = globalThreadIdx[2]; i < *modulesInGPU.nLowerModules; i += gridThreadExtent[2]) {
        if (segmentsInGPU.nSegments[i] == 0) {
          rangesInGPU.segmentRanges[i * 2] = -1;
          rangesInGPU.segmentRanges[i * 2 + 1] = -1;
        } else {
          rangesInGPU.segmentRanges[i * 2] = rangesInGPU.segmentModuleIndices[i];
          rangesInGPU.segmentRanges[i * 2 + 1] = rangesInGPU.segmentModuleIndices[i] + segmentsInGPU.nSegments[i] - 1;
        }
      }
    }
  };

  struct addPixelSegmentToEventKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  struct SDL::modules modulesInGPU,
                                  struct SDL::objectRanges rangesInGPU,
                                  struct SDL::hits hitsInGPU,
                                  struct SDL::miniDoublets mdsInGPU,
                                  struct SDL::segments segmentsInGPU,
                                  unsigned int* hitIndices0,
                                  unsigned int* hitIndices1,
                                  unsigned int* hitIndices2,
                                  unsigned int* hitIndices3,
                                  float* dPhiChange,
                                  uint16_t pixelModuleIndex,
                                  const int size) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (int tid = globalThreadIdx[2]; tid < size; tid += gridThreadExtent[2]) {
        unsigned int innerMDIndex = rangesInGPU.miniDoubletModuleIndices[pixelModuleIndex] + 2 * (tid);
        unsigned int outerMDIndex = rangesInGPU.miniDoubletModuleIndices[pixelModuleIndex] + 2 * (tid) + 1;
        unsigned int pixelSegmentIndex = rangesInGPU.segmentModuleIndices[pixelModuleIndex] + tid;

        addMDToMemory(acc,
                      mdsInGPU,
                      hitsInGPU,
                      modulesInGPU,
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
                      0,
                      innerMDIndex);
        addMDToMemory(acc,
                      mdsInGPU,
                      hitsInGPU,
                      modulesInGPU,
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
                      0,
                      outerMDIndex);

        //in outer hits - pt, eta, phi
        float slope = alpaka::math::sinh(acc, hitsInGPU.ys[mdsInGPU.outerHitIndices[innerMDIndex]]);
        float intercept = hitsInGPU.zs[mdsInGPU.anchorHitIndices[innerMDIndex]] -
                          slope * hitsInGPU.rts[mdsInGPU.anchorHitIndices[innerMDIndex]];
        float score_lsq = (hitsInGPU.rts[mdsInGPU.anchorHitIndices[outerMDIndex]] * slope + intercept) -
                          (hitsInGPU.zs[mdsInGPU.anchorHitIndices[outerMDIndex]]);
        score_lsq = score_lsq * score_lsq;

        unsigned int hits1[4];
        hits1[0] = hitsInGPU.idxs[mdsInGPU.anchorHitIndices[innerMDIndex]];
        hits1[1] = hitsInGPU.idxs[mdsInGPU.anchorHitIndices[outerMDIndex]];
        hits1[2] = hitsInGPU.idxs[mdsInGPU.outerHitIndices[innerMDIndex]];
        hits1[3] = hitsInGPU.idxs[mdsInGPU.outerHitIndices[outerMDIndex]];
        addPixelSegmentToMemory(acc,
                                segmentsInGPU,
                                mdsInGPU,
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
}  // namespace SDL

#endif
