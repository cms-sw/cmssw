#ifndef RecoTracker_LSTCore_src_alpaka_Segment_h
#define RecoTracker_LSTCore_src_alpaka_Segment_h

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/Module.h"
#include "RecoTracker/LSTCore/interface/EndcapGeometry.h"

#include "MiniDoublet.h"
#include "Hit.h"
#include "ObjectRanges.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  struct Segments {
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
    PixelType* pixelType;
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
    void setData(TBuff& buf) {
      dPhis = buf.dPhis_buf.data();
      dPhiMins = buf.dPhiMins_buf.data();
      dPhiMaxs = buf.dPhiMaxs_buf.data();
      dPhiChanges = buf.dPhiChanges_buf.data();
      dPhiChangeMins = buf.dPhiChangeMins_buf.data();
      dPhiChangeMaxs = buf.dPhiChangeMaxs_buf.data();
      innerLowerModuleIndices = buf.innerLowerModuleIndices_buf.data();
      outerLowerModuleIndices = buf.outerLowerModuleIndices_buf.data();
      seedIdx = buf.seedIdx_buf.data();
      mdIndices = buf.mdIndices_buf.data();
      nMemoryLocations = buf.nMemoryLocations_buf.data();
      innerMiniDoubletAnchorHitIndices = buf.innerMiniDoubletAnchorHitIndices_buf.data();
      outerMiniDoubletAnchorHitIndices = buf.outerMiniDoubletAnchorHitIndices_buf.data();
      charge = buf.charge_buf.data();
      superbin = buf.superbin_buf.data();
      nSegments = buf.nSegments_buf.data();
      totOccupancySegments = buf.totOccupancySegments_buf.data();
      pLSHitsIdxs = buf.pLSHitsIdxs_buf.data();
      pixelType = buf.pixelType_buf.data();
      isQuad = buf.isQuad_buf.data();
      isDup = buf.isDup_buf.data();
      partOfPT5 = buf.partOfPT5_buf.data();
      ptIn = buf.ptIn_buf.data();
      ptErr = buf.ptErr_buf.data();
      px = buf.px_buf.data();
      py = buf.py_buf.data();
      pz = buf.pz_buf.data();
      etaErr = buf.etaErr_buf.data();
      eta = buf.eta_buf.data();
      phi = buf.phi_buf.data();
      score = buf.score_buf.data();
      circleCenterX = buf.circleCenterX_buf.data();
      circleCenterY = buf.circleCenterY_buf.data();
      circleRadius = buf.circleRadius_buf.data();
    }
  };

  template <typename TDev>
  struct SegmentsBuffer {
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
    Buf<TDev, PixelType> pixelType_buf;
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

    Segments data_;

    template <typename TQueue, typename TDevAcc>
    SegmentsBuffer(unsigned int nMemoryLocationsIn,
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
          pixelType_buf(allocBufWrapper<PixelType>(devAccIn, maxPixelSegments, queue)),
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
    }

    inline Segments const* data() const { return &data_; }
    inline void setData(SegmentsBuffer& buf) { data_.setData(buf); }
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float isTighterTiltedModules_seg(Modules const& modulesInGPU,
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
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float isTighterTiltedModules_seg(short subdet, short layer, short side, short rod) {
    // The "tighter" tilted modules are the subset of tilted modules that have smaller spacing
    // This is the same as what was previously considered as"isNormalTiltedModules"
    // See Figure 9.1 of https://cds.cern.ch/record/2272264/files/CMS-TDR-014.pdf
    return (subdet == Barrel) && (((side != Center) && (layer == 3)) ||
                                  ((side == NegZ) && (((layer == 2) && (rod > 5)) || ((layer == 1) && (rod > 9)))) ||
                                  ((side == PosZ) && (((layer == 2) && (rod < 8)) || ((layer == 1) && (rod < 4)))));
  }

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
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float moduleGapSize_seg(Modules const& modulesInGPU, unsigned int moduleIndex) {
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
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void dAlphaThreshold(TAcc const& acc,
                                                      float* dAlphaThresholdValues,
                                                      Modules const& modulesInGPU,
                                                      MiniDoublets const& mdsInGPU,
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
    float sdMuls = (modulesInGPU.subdets[innerLowerModuleIndex] == Barrel)
                       ? kMiniMulsPtScaleBarrel[modulesInGPU.layers[innerLowerModuleIndex] - 1] * 3.f / ptCut
                       : kMiniMulsPtScaleEndcap[modulesInGPU.layers[innerLowerModuleIndex] - 1] * 3.f / ptCut;

    //more accurate then outer rt - inner rt
    float segmentDr = alpaka::math::sqrt(acc, (yOut - yIn) * (yOut - yIn) + (xOut - xIn) * (xOut - xIn));

    const float dAlpha_Bfield =
        alpaka::math::asin(acc, alpaka::math::min(acc, segmentDr * k2Rinv1GeVf / ptCut, kSinAlphaMax));

    bool isInnerTilted =
        modulesInGPU.subdets[innerLowerModuleIndex] == Barrel and modulesInGPU.sides[innerLowerModuleIndex] != Center;
    bool isOuterTilted =
        modulesInGPU.subdets[outerLowerModuleIndex] == Barrel and modulesInGPU.sides[outerLowerModuleIndex] != Center;

    float drdzInner = modulesInGPU.drdzs[innerLowerModuleIndex];
    float drdzOuter = modulesInGPU.drdzs[outerLowerModuleIndex];
    float innerModuleGapSize = moduleGapSize_seg(modulesInGPU, innerLowerModuleIndex);
    float outerModuleGapSize = moduleGapSize_seg(modulesInGPU, outerLowerModuleIndex);
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

    if (modulesInGPU.subdets[innerLowerModuleIndex] == Barrel) {
      sdLumForInnerMini2 = innerminiTilt2 * (dAlpha_Bfield * dAlpha_Bfield);
    } else {
      sdLumForInnerMini2 = (mdsInGPU.dphis[innerMDIndex] * mdsInGPU.dphis[innerMDIndex]) * (kDeltaZLum * kDeltaZLum) /
                           (mdsInGPU.dzs[innerMDIndex] * mdsInGPU.dzs[innerMDIndex]);
    }

    if (modulesInGPU.subdets[outerLowerModuleIndex] == Barrel) {
      sdLumForOuterMini2 = outerminiTilt2 * (dAlpha_Bfield * dAlpha_Bfield);
    } else {
      sdLumForOuterMini2 = (mdsInGPU.dphis[outerMDIndex] * mdsInGPU.dphis[outerMDIndex]) * (kDeltaZLum * kDeltaZLum) /
                           (mdsInGPU.dzs[outerMDIndex] * mdsInGPU.dzs[outerMDIndex]);
    }

    // Unique stuff for the segment dudes alone
    float dAlpha_res_inner =
        0.02f / miniDelta *
        (modulesInGPU.subdets[innerLowerModuleIndex] == Barrel ? 1.0f : alpaka::math::abs(acc, zIn) / rtIn);
    float dAlpha_res_outer =
        0.02f / miniDelta *
        (modulesInGPU.subdets[outerLowerModuleIndex] == Barrel ? 1.0f : alpaka::math::abs(acc, zOut) / rtOut);

    float dAlpha_res = dAlpha_res_inner + dAlpha_res_outer;

    if (modulesInGPU.subdets[innerLowerModuleIndex] == Barrel and modulesInGPU.sides[innerLowerModuleIndex] == Center) {
      dAlphaThresholdValues[0] = dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls);
    } else {
      dAlphaThresholdValues[0] =
          dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls + sdLumForInnerMini2);
    }

    if (modulesInGPU.subdets[outerLowerModuleIndex] == Barrel and modulesInGPU.sides[outerLowerModuleIndex] == Center) {
      dAlphaThresholdValues[1] = dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls);
    } else {
      dAlphaThresholdValues[1] =
          dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls + sdLumForOuterMini2);
    }

    //Inner to outer
    dAlphaThresholdValues[2] = dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls);
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addSegmentToMemory(Segments& segmentsInGPU,
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
                                                              Segments& segmentsInGPU,
                                                              MiniDoublets const& mdsInGPU,
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
    float bestChiSquared = lst_INF;
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
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runSegmentDefaultAlgoBarrel(TAcc const& acc,
                                                                  Modules const& modulesInGPU,
                                                                  MiniDoublets const& mdsInGPU,
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
    float sdMuls = (modulesInGPU.subdets[innerLowerModuleIndex] == Barrel)
                       ? kMiniMulsPtScaleBarrel[modulesInGPU.layers[innerLowerModuleIndex] - 1] * 3.f / ptCut
                       : kMiniMulsPtScaleEndcap[modulesInGPU.layers[innerLowerModuleIndex] - 1] * 3.f / ptCut;

    float xIn, yIn, zIn, rtIn, xOut, yOut, zOut, rtOut;

    xIn = mdsInGPU.anchorX[innerMDIndex];
    yIn = mdsInGPU.anchorY[innerMDIndex];
    zIn = mdsInGPU.anchorZ[innerMDIndex];
    rtIn = mdsInGPU.anchorRt[innerMDIndex];

    xOut = mdsInGPU.anchorX[outerMDIndex];
    yOut = mdsInGPU.anchorY[outerMDIndex];
    zOut = mdsInGPU.anchorZ[outerMDIndex];
    rtOut = mdsInGPU.anchorRt[outerMDIndex];

    float sdSlope = alpaka::math::asin(acc, alpaka::math::min(acc, rtOut * k2Rinv1GeVf / ptCut, kSinAlphaMax));
    float sdPVoff = 0.1f / rtOut;
    float dzDrtScale = alpaka::math::tan(acc, sdSlope) / sdSlope;  //FIXME: need appropriate value

    const float zGeom = modulesInGPU.layers[innerLowerModuleIndex] <= 2 ? 2.f * kPixelPSZpitch : 2.f * kStrip2SZpitch;

    float zLo = zIn + (zIn - kDeltaZLum) * (rtOut / rtIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) -
                zGeom;  //slope-correction only on outer end
    float zHi = zIn + (zIn + kDeltaZLum) * (rtOut / rtIn - 1.f) * (zIn < 0.f ? 1.f : dzDrtScale) + zGeom;

    if ((zOut < zLo) || (zOut > zHi))
      return false;

    float sdCut = sdSlope + alpaka::math::sqrt(acc, sdMuls * sdMuls + sdPVoff * sdPVoff);

    dPhi = phi_mpi_pi(acc, mdsInGPU.anchorPhi[outerMDIndex] - mdsInGPU.anchorPhi[innerMDIndex]);

    if (alpaka::math::abs(acc, dPhi) > sdCut)
      return false;

    dPhiChange = phi_mpi_pi(acc, phi(acc, xOut - xIn, yOut - yIn) - mdsInGPU.anchorPhi[innerMDIndex]);

    if (alpaka::math::abs(acc, dPhiChange) > sdCut)
      return false;

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
                                                                  Modules const& modulesInGPU,
                                                                  MiniDoublets const& mdsInGPU,
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

    xIn = mdsInGPU.anchorX[innerMDIndex];
    yIn = mdsInGPU.anchorY[innerMDIndex];
    zIn = mdsInGPU.anchorZ[innerMDIndex];
    rtIn = mdsInGPU.anchorRt[innerMDIndex];

    xOut = mdsInGPU.anchorX[outerMDIndex];
    yOut = mdsInGPU.anchorY[outerMDIndex];
    zOut = mdsInGPU.anchorZ[outerMDIndex];
    rtOut = mdsInGPU.anchorRt[outerMDIndex];

    bool outerLayerEndcapTwoS = (modulesInGPU.subdets[outerLowerModuleIndex] == Endcap) &&
                                (modulesInGPU.moduleType[outerLowerModuleIndex] == TwoS);

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
    // Alpaka: Needs to be moved over
    float dLum = alpaka::math::copysign(acc, kDeltaZLum, zIn);
    float drtDzScale = sdSlope / alpaka::math::tan(acc, sdSlope);

    float rtLo = alpaka::math::max(
        acc, rtIn * (1.f + dz / (zIn + dLum) * drtDzScale) - rtGeom, rtIn - 0.5f * rtGeom);  //rt should increase
    float rtHi = rtIn * (zOut - dLum) / (zIn - dLum) +
                 rtGeom;  //dLum for luminous; rGeom for measurement size; no tanTheta_loc(pt) correction

    // Completeness
    if ((rtOut < rtLo) || (rtOut > rtHi))
      return false;

    dPhi = phi_mpi_pi(acc, mdsInGPU.anchorPhi[outerMDIndex] - mdsInGPU.anchorPhi[innerMDIndex]);

    float sdCut = sdSlope;
    if (outerLayerEndcapTwoS) {
      float dPhiPos_high = phi_mpi_pi(acc, mdsInGPU.anchorHighEdgePhi[outerMDIndex] - mdsInGPU.anchorPhi[innerMDIndex]);
      float dPhiPos_low = phi_mpi_pi(acc, mdsInGPU.anchorLowEdgePhi[outerMDIndex] - mdsInGPU.anchorPhi[innerMDIndex]);

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
                                                            Modules const& modulesInGPU,
                                                            MiniDoublets const& mdsInGPU,
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
    if (modulesInGPU.subdets[innerLowerModuleIndex] == Barrel and
        modulesInGPU.subdets[outerLowerModuleIndex] == Barrel) {
      return runSegmentDefaultAlgoBarrel(acc,
                                         modulesInGPU,
                                         mdsInGPU,
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
                                         modulesInGPU,
                                         mdsInGPU,
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

  struct CreateSegmentsInGPUv2 {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  Modules modulesInGPU,
                                  MiniDoublets mdsInGPU,
                                  Segments segmentsInGPU,
                                  ObjectRanges rangesInGPU) const {
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
              modulesInGPU.moduleMap[innerLowerModuleIndex * max_connected_modules + outerLowerModuleArrayIdx];

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

            float dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax;

            unsigned int innerMiniDoubletAnchorHitIndex = mdsInGPU.anchorHitIndices[innerMDIndex];
            unsigned int outerMiniDoubletAnchorHitIndex = mdsInGPU.anchorHitIndices[outerMDIndex];
            dPhiMin = 0;
            dPhiMax = 0;
            dPhiChangeMin = 0;
            dPhiChangeMax = 0;
            if (runSegmentDefaultAlgo(acc,
                                      modulesInGPU,
                                      mdsInGPU,
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
              unsigned int totOccupancySegments = alpaka::atomicAdd(
                  acc, &segmentsInGPU.totOccupancySegments[innerLowerModuleIndex], 1u, alpaka::hierarchy::Threads{});
              if (static_cast<int>(totOccupancySegments) >= rangesInGPU.segmentModuleOccupancy[innerLowerModuleIndex]) {
#ifdef WARNINGS
                printf("Segment excess alert! Module index = %d\n", innerLowerModuleIndex);
#endif
              } else {
                unsigned int segmentModuleIdx = alpaka::atomicAdd(
                    acc, &segmentsInGPU.nSegments[innerLowerModuleIndex], 1u, alpaka::hierarchy::Threads{});
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

  struct CreateSegmentArrayRanges {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  Modules modulesInGPU,
                                  ObjectRanges rangesInGPU,
                                  MiniDoublets mdsInGPU) const {
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

      // Create variables outside of the for loop.
      int occupancy, category_number, eta_number;

      for (uint16_t i = globalThreadIdx[0]; i < *modulesInGPU.nLowerModules; i += gridThreadExtent[0]) {
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
        rangesInGPU.segmentModuleIndices[i] = nTotSegs;
        rangesInGPU.segmentModuleOccupancy[i] = occupancy;
      }

      // Wait for all threads to finish before reporting final values
      alpaka::syncBlockThreads(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        rangesInGPU.segmentModuleIndices[*modulesInGPU.nLowerModules] = nTotalSegments;
        *rangesInGPU.device_nTotalSegs = nTotalSegments;
      }
    }
  };

  struct AddSegmentRangesToEventExplicit {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  Modules modulesInGPU,
                                  Segments segmentsInGPU,
                                  ObjectRanges rangesInGPU) const {
      // implementation is 1D with a single block
      static_assert(std::is_same_v<TAcc, ALPAKA_ACCELERATOR_NAMESPACE::Acc1D>, "Should be Acc1D");
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (uint16_t i = globalThreadIdx[0]; i < *modulesInGPU.nLowerModules; i += gridThreadExtent[0]) {
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

  struct AddPixelSegmentToEventKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  Modules modulesInGPU,
                                  ObjectRanges rangesInGPU,
                                  Hits hitsInGPU,
                                  MiniDoublets mdsInGPU,
                                  Segments segmentsInGPU,
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
                      outerMDIndex);

        //in outer hits - pt, eta, phi
        float slope = alpaka::math::sinh(acc, hitsInGPU.ys[mdsInGPU.outerHitIndices[innerMDIndex]]);
        float intercept = hitsInGPU.zs[mdsInGPU.anchorHitIndices[innerMDIndex]] -
                          slope * hitsInGPU.rts[mdsInGPU.anchorHitIndices[innerMDIndex]];
        float score_lsq = (hitsInGPU.rts[mdsInGPU.anchorHitIndices[outerMDIndex]] * slope + intercept) -
                          (hitsInGPU.zs[mdsInGPU.anchorHitIndices[outerMDIndex]]);
        score_lsq = score_lsq * score_lsq;

        unsigned int hits1[Params_pLS::kHits];
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
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif
