#ifndef RecoTracker_LSTCore_src_alpaka_Triplet_h
#define RecoTracker_LSTCore_src_alpaka_Triplet_h

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/Module.h"

#include "Segment.h"
#include "MiniDoublet.h"
#include "Hit.h"
#include "ObjectRanges.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  struct Triplets {
    unsigned int* segmentIndices;
    uint16_t* lowerModuleIndices;  //3 of them
    unsigned int* nTriplets;
    unsigned int* totOccupancyTriplets;
    unsigned int* nMemoryLocations;
    uint8_t* logicalLayers;
    unsigned int* hitIndices;
    FPX* betaIn;
    float* circleRadius;
    float* circleCenterX;
    float* circleCenterY;
    bool* partOfPT5;
    bool* partOfT5;
    bool* partOfPT3;

#ifdef CUT_VALUE_DEBUG
    //debug variables
    float* zOut;
    float* rtOut;
    float* betaInCut;
#endif
    template <typename TBuff>
    void setData(TBuff& buf) {
      segmentIndices = buf.segmentIndices_buf.data();
      lowerModuleIndices = buf.lowerModuleIndices_buf.data();
      nTriplets = buf.nTriplets_buf.data();
      totOccupancyTriplets = buf.totOccupancyTriplets_buf.data();
      nMemoryLocations = buf.nMemoryLocations_buf.data();
      logicalLayers = buf.logicalLayers_buf.data();
      hitIndices = buf.hitIndices_buf.data();
      betaIn = buf.betaIn_buf.data();
      circleRadius = buf.circleRadius_buf.data();
      circleCenterX = buf.circleCenterX_buf.data();
      circleCenterY = buf.circleCenterY_buf.data();
      partOfPT5 = buf.partOfPT5_buf.data();
      partOfT5 = buf.partOfT5_buf.data();
      partOfPT3 = buf.partOfPT3_buf.data();
#ifdef CUT_VALUE_DEBUG
      zOut = buf.zOut_buf.data();
      rtOut = buf.rtOut_buf.data();
      betaInCut = buf.betaInCut_buf.data();
#endif
    }
  };

  template <typename TDev>
  struct TripletsBuffer {
    Buf<TDev, unsigned int> segmentIndices_buf;
    Buf<TDev, uint16_t> lowerModuleIndices_buf;
    Buf<TDev, unsigned int> nTriplets_buf;
    Buf<TDev, unsigned int> totOccupancyTriplets_buf;
    Buf<TDev, unsigned int> nMemoryLocations_buf;
    Buf<TDev, uint8_t> logicalLayers_buf;
    Buf<TDev, unsigned int> hitIndices_buf;
    Buf<TDev, FPX> betaIn_buf;
    Buf<TDev, float> circleRadius_buf;
    Buf<TDev, float> circleCenterX_buf;
    Buf<TDev, float> circleCenterY_buf;
    Buf<TDev, bool> partOfPT5_buf;
    Buf<TDev, bool> partOfT5_buf;
    Buf<TDev, bool> partOfPT3_buf;

#ifdef CUT_VALUE_DEBUG
    Buf<TDev, float> zOut_buf;
    Buf<TDev, float> rtOut_buf;
    Buf<TDev, float> deltaPhiPos_buf;
    Buf<TDev, float> deltaPhi_buf;
    Buf<TDev, float> zLo_buf;
    Buf<TDev, float> zHi_buf;
    Buf<TDev, float> zLoPointed_buf;
    Buf<TDev, float> zHiPointed_buf;
    Buf<TDev, float> dPhiCut_buf;
    Buf<TDev, float> betaInCut_buf;
    Buf<TDev, float> rtLo_buf;
    Buf<TDev, float> rtHi_buf;
#endif

    Triplets data_;

    template <typename TQueue, typename TDevAcc>
    TripletsBuffer(unsigned int maxTriplets, unsigned int nLowerModules, TDevAcc const& devAccIn, TQueue& queue)
        : segmentIndices_buf(allocBufWrapper<unsigned int>(devAccIn, 2 * maxTriplets, queue)),
          lowerModuleIndices_buf(allocBufWrapper<uint16_t>(devAccIn, Params_T3::kLayers * maxTriplets, queue)),
          nTriplets_buf(allocBufWrapper<unsigned int>(devAccIn, nLowerModules, queue)),
          totOccupancyTriplets_buf(allocBufWrapper<unsigned int>(devAccIn, nLowerModules, queue)),
          nMemoryLocations_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          logicalLayers_buf(allocBufWrapper<uint8_t>(devAccIn, maxTriplets * Params_T3::kLayers, queue)),
          hitIndices_buf(allocBufWrapper<unsigned int>(devAccIn, maxTriplets * Params_T3::kHits, queue)),
          betaIn_buf(allocBufWrapper<FPX>(devAccIn, maxTriplets, queue)),
          circleRadius_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          circleCenterX_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          circleCenterY_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          partOfPT5_buf(allocBufWrapper<bool>(devAccIn, maxTriplets, queue)),
          partOfT5_buf(allocBufWrapper<bool>(devAccIn, maxTriplets, queue)),
          partOfPT3_buf(allocBufWrapper<bool>(devAccIn, maxTriplets, queue))
#ifdef CUT_VALUE_DEBUG
          ,
          zOut_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          rtOut_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          deltaPhiPos_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          deltaPhi_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          zLo_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          zHi_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          zLoPointed_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          zHiPointed_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          dPhiCut_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          betaInCut_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          rtLo_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          rtHi_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue))
#endif
    {
      alpaka::memset(queue, nTriplets_buf, 0u);
      alpaka::memset(queue, totOccupancyTriplets_buf, 0u);
      alpaka::memset(queue, partOfPT5_buf, false);
      alpaka::memset(queue, partOfT5_buf, false);
      alpaka::memset(queue, partOfPT3_buf, false);
    }

    inline Triplets const* data() const { return &data_; }
    inline void setData(TripletsBuffer& buf) { data_.setData(buf); }
  };

#ifdef CUT_VALUE_DEBUG
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addTripletToMemory(Modules const& modulesInGPU,
                                                         MiniDoublets const& mdsInGPU,
                                                         Segments const& segmentsInGPU,
                                                         Triplets& tripletsInGPU,
                                                         unsigned int innerSegmentIndex,
                                                         unsigned int outerSegmentIndex,
                                                         uint16_t innerInnerLowerModuleIndex,
                                                         uint16_t middleLowerModuleIndex,
                                                         uint16_t outerOuterLowerModuleIndex,
                                                         float zOut,
                                                         float rtOut,
                                                         float betaIn,
                                                         float betaInCut,
                                                         float circleRadius,
                                                         float circleCenterX,
                                                         float circleCenterY,
                                                         unsigned int tripletIndex)
#else
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addTripletToMemory(Modules const& modulesInGPU,
                                                         MiniDoublets const& mdsInGPU,
                                                         Segments const& segmentsInGPU,
                                                         Triplets& tripletsInGPU,
                                                         unsigned int innerSegmentIndex,
                                                         unsigned int outerSegmentIndex,
                                                         uint16_t innerInnerLowerModuleIndex,
                                                         uint16_t middleLowerModuleIndex,
                                                         uint16_t outerOuterLowerModuleIndex,
                                                         float betaIn,
                                                         float circleRadius,
                                                         float circleCenterX,
                                                         float circleCenterY,
                                                         unsigned int tripletIndex)
#endif
  {
    tripletsInGPU.segmentIndices[tripletIndex * 2] = innerSegmentIndex;
    tripletsInGPU.segmentIndices[tripletIndex * 2 + 1] = outerSegmentIndex;
    tripletsInGPU.lowerModuleIndices[tripletIndex * Params_T3::kLayers] = innerInnerLowerModuleIndex;
    tripletsInGPU.lowerModuleIndices[tripletIndex * Params_T3::kLayers + 1] = middleLowerModuleIndex;
    tripletsInGPU.lowerModuleIndices[tripletIndex * Params_T3::kLayers + 2] = outerOuterLowerModuleIndex;

    tripletsInGPU.betaIn[tripletIndex] = __F2H(betaIn);
    tripletsInGPU.circleRadius[tripletIndex] = circleRadius;
    tripletsInGPU.circleCenterX[tripletIndex] = circleCenterX;
    tripletsInGPU.circleCenterY[tripletIndex] = circleCenterY;
    tripletsInGPU.logicalLayers[tripletIndex * Params_T3::kLayers] =
        modulesInGPU.layers[innerInnerLowerModuleIndex] + (modulesInGPU.subdets[innerInnerLowerModuleIndex] == 4) * 6;
    tripletsInGPU.logicalLayers[tripletIndex * Params_T3::kLayers + 1] =
        modulesInGPU.layers[middleLowerModuleIndex] + (modulesInGPU.subdets[middleLowerModuleIndex] == 4) * 6;
    tripletsInGPU.logicalLayers[tripletIndex * Params_T3::kLayers + 2] =
        modulesInGPU.layers[outerOuterLowerModuleIndex] + (modulesInGPU.subdets[outerOuterLowerModuleIndex] == 4) * 6;
    //get the hits
    unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex];
    unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex + 1];
    unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex + 1];

    tripletsInGPU.hitIndices[tripletIndex * Params_T3::kHits] = mdsInGPU.anchorHitIndices[firstMDIndex];
    tripletsInGPU.hitIndices[tripletIndex * Params_T3::kHits + 1] = mdsInGPU.outerHitIndices[firstMDIndex];
    tripletsInGPU.hitIndices[tripletIndex * Params_T3::kHits + 2] = mdsInGPU.anchorHitIndices[secondMDIndex];
    tripletsInGPU.hitIndices[tripletIndex * Params_T3::kHits + 3] = mdsInGPU.outerHitIndices[secondMDIndex];
    tripletsInGPU.hitIndices[tripletIndex * Params_T3::kHits + 4] = mdsInGPU.anchorHitIndices[thirdMDIndex];
    tripletsInGPU.hitIndices[tripletIndex * Params_T3::kHits + 5] = mdsInGPU.outerHitIndices[thirdMDIndex];
#ifdef CUT_VALUE_DEBUG
    tripletsInGPU.zOut[tripletIndex] = zOut;
    tripletsInGPU.rtOut[tripletIndex] = rtOut;
    tripletsInGPU.betaInCut[tripletIndex] = betaInCut;
#endif
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passRZConstraint(TAcc const& acc,
                                                       Modules const& modulesInGPU,
                                                       MiniDoublets const& mdsInGPU,
                                                       Segments const& segmentsInGPU,
                                                       uint16_t innerInnerLowerModuleIndex,
                                                       uint16_t middleLowerModuleIndex,
                                                       uint16_t outerOuterLowerModuleIndex,
                                                       unsigned int firstMDIndex,
                                                       unsigned int secondMDIndex,
                                                       unsigned int thirdMDIndex) {
    //get the rt and z
    const float& r1 = mdsInGPU.anchorRt[firstMDIndex];
    const float& r2 = mdsInGPU.anchorRt[secondMDIndex];
    const float& r3 = mdsInGPU.anchorRt[thirdMDIndex];

    const float& z1 = mdsInGPU.anchorZ[firstMDIndex];
    const float& z2 = mdsInGPU.anchorZ[secondMDIndex];
    const float& z3 = mdsInGPU.anchorZ[thirdMDIndex];

    // Using lst_layer numbering convention defined in ModuleMethods.h
    const int layer1 = modulesInGPU.lstLayers[innerInnerLowerModuleIndex];
    const int layer2 = modulesInGPU.lstLayers[middleLowerModuleIndex];
    const int layer3 = modulesInGPU.lstLayers[outerOuterLowerModuleIndex];

    const float residual = z2 - ((z3 - z1) / (r3 - r1) * (r2 - r1) + z1);

    if (layer1 == 12 and layer2 == 13 and layer3 == 14) {
      return false;
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 3) {
      return alpaka::math::abs(acc, residual) < 0.53f;
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 7) {
      return alpaka::math::abs(acc, residual) < 1;
    } else if (layer1 == 13 and layer2 == 14 and layer3 == 15) {
      return false;
    } else if (layer1 == 14 and layer2 == 15 and layer3 == 16) {
      return false;
    } else if (layer1 == 1 and layer2 == 7 and layer3 == 8) {
      return alpaka::math::abs(acc, residual) < 1;
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 4) {
      return alpaka::math::abs(acc, residual) < 1.21f;
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 7) {
      return alpaka::math::abs(acc, residual) < 1.f;
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 8) {
      return alpaka::math::abs(acc, residual) < 1.f;
    } else if (layer1 == 3 and layer2 == 4 and layer3 == 5) {
      return alpaka::math::abs(acc, residual) < 2.7f;
    } else if (layer1 == 4 and layer2 == 5 and layer3 == 6) {
      return alpaka::math::abs(acc, residual) < 3.06f;
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 9) {
      return alpaka::math::abs(acc, residual) < 1;
    } else if (layer1 == 8 and layer2 == 9 and layer3 == 10) {
      return alpaka::math::abs(acc, residual) < 1;
    } else if (layer1 == 9 and layer2 == 10 and layer3 == 11) {
      return alpaka::math::abs(acc, residual) < 1;
    } else {
      return alpaka::math::abs(acc, residual) < 5;
    }
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPointingConstraintBBB(TAcc const& acc,
                                                                Modules const& modulesInGPU,
                                                                MiniDoublets const& mdsInGPU,
                                                                Segments const& segmentsInGPU,
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
                                                                float& betaInCut) {
    bool isPSIn = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == PS);
    bool isPSOut = (modulesInGPU.moduleType[outerOuterLowerModuleIndex] == PS);

    float rtIn = mdsInGPU.anchorRt[firstMDIndex];
    float rtMid = mdsInGPU.anchorRt[secondMDIndex];
    rtOut = mdsInGPU.anchorRt[thirdMDIndex];

    float zIn = mdsInGPU.anchorZ[firstMDIndex];
    float zMid = mdsInGPU.anchorZ[secondMDIndex];
    zOut = mdsInGPU.anchorZ[thirdMDIndex];

    float alpha1GeVOut = alpaka::math::asin(acc, alpaka::math::min(acc, rtOut * k2Rinv1GeVf / ptCut, kSinAlphaMax));

    float rtRatio_OutIn = rtOut / rtIn;  // Outer segment beginning rt divided by inner segment beginning rt;
    float dzDrtScale = alpaka::math::tan(acc, alpha1GeVOut) / alpha1GeVOut;  // The track can bend in r-z plane slightly
    float zpitchIn = (isPSIn ? kPixelPSZpitch : kStrip2SZpitch);
    float zpitchOut = (isPSOut ? kPixelPSZpitch : kStrip2SZpitch);

    const float zHi =
        zIn + (zIn + kDeltaZLum) * (rtRatio_OutIn - 1.f) * (zIn < 0.f ? 1.f : dzDrtScale) + (zpitchIn + zpitchOut);
    const float zLo = zIn + (zIn - kDeltaZLum) * (rtRatio_OutIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) -
                      (zpitchIn + zpitchOut);  //slope-correction only on outer end

    //Cut 1 - z compatibility
    if ((zOut < zLo) || (zOut > zHi))
      return false;

    float drt_OutIn = (rtOut - rtIn);

    float r3In = alpaka::math::sqrt(acc, zIn * zIn + rtIn * rtIn);
    float drt_InSeg = rtMid - rtIn;
    float dz_InSeg = zMid - zIn;
    float dr3_InSeg =
        alpaka::math::sqrt(acc, rtMid * rtMid + zMid * zMid) - alpaka::math::sqrt(acc, rtIn * rtIn + zIn * zIn);

    float coshEta = dr3_InSeg / drt_InSeg;
    float dzErr = (zpitchIn + zpitchOut) * (zpitchIn + zpitchOut) * 2.f;

    float thetaMuls2 = (kMulsInGeV * kMulsInGeV) * (0.1f + 0.2f * (rtOut - rtIn) / 50.f) * (r3In / rtIn);
    float muls2 = thetaMuls2 * 9.f / (ptCut * ptCut) * 16.f;
    dzErr += muls2 * drt_OutIn * drt_OutIn / 3.f * coshEta * coshEta;
    dzErr = alpaka::math::sqrt(acc, dzErr);

    // Constructing upper and lower bound
    const float dzMean = dz_InSeg / drt_InSeg * drt_OutIn;
    const float zWindow = dzErr / drt_InSeg * drt_OutIn +
                          (zpitchIn + zpitchOut);  //FIXME for ptCut lower than ~0.8 need to add curv path correction
    const float zLoPointed = zIn + dzMean * (zIn > 0.f ? 1.f : dzDrtScale) - zWindow;
    const float zHiPointed = zIn + dzMean * (zIn < 0.f ? 1.f : dzDrtScale) + zWindow;

    // Constructing upper and lower bound

    // Cut #2: Pointed Z (Inner segment two MD points to outer segment inner MD)
    if ((zOut < zLoPointed) || (zOut > zHiPointed))
      return false;

    // raw betaIn value without any correction, based on the mini-doublet hit positions
    float alpha_InLo = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
    float tl_axis_x = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float tl_axis_y = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];
    betaIn = alpha_InLo - phi_mpi_pi(acc, phi(acc, tl_axis_x, tl_axis_y) - mdsInGPU.anchorPhi[firstMDIndex]);

    //beta computation
    float drt_tl_axis = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);

    //innerOuterAnchor - innerInnerAnchor
    const float rt_InSeg =
        alpaka::math::sqrt(acc,
                           (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) *
                                   (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) +
                               (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]) *
                                   (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]));
    betaInCut =
        alpaka::math::asin(acc, alpaka::math::min(acc, (-rt_InSeg + drt_tl_axis) * k2Rinv1GeVf / ptCut, kSinAlphaMax)) +
        (0.02f / drt_InSeg);

    //Cut #3: first beta cut
    return alpaka::math::abs(acc, betaIn) < betaInCut;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPointingConstraintBBE(TAcc const& acc,
                                                                Modules const& modulesInGPU,
                                                                MiniDoublets const& mdsInGPU,
                                                                Segments const& segmentsInGPU,
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
                                                                float& betaInCut) {
    bool isPSIn = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == PS);
    bool isPSOut = (modulesInGPU.moduleType[outerOuterLowerModuleIndex] == PS);

    float rtIn = mdsInGPU.anchorRt[firstMDIndex];
    float rtMid = mdsInGPU.anchorRt[secondMDIndex];
    rtOut = mdsInGPU.anchorRt[thirdMDIndex];

    float zIn = mdsInGPU.anchorZ[firstMDIndex];
    float zMid = mdsInGPU.anchorZ[secondMDIndex];
    zOut = mdsInGPU.anchorZ[thirdMDIndex];

    float alpha1GeV_OutLo = alpaka::math::asin(acc, alpaka::math::min(acc, rtOut * k2Rinv1GeVf / ptCut, kSinAlphaMax));

    float dzDrtScale =
        alpaka::math::tan(acc, alpha1GeV_OutLo) / alpha1GeV_OutLo;  // The track can bend in r-z plane slightly
    float zpitchIn = (isPSIn ? kPixelPSZpitch : kStrip2SZpitch);
    float zpitchOut = (isPSOut ? kPixelPSZpitch : kStrip2SZpitch);
    float zGeom = zpitchIn + zpitchOut;

    // Cut #0: Preliminary (Only here in endcap case)
    if (zIn * zOut <= 0)
      return false;

    float dLum = alpaka::math::copysign(acc, kDeltaZLum, zIn);
    bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerOuterLowerModuleIndex] == PS;
    float rtGeom1 = isOutSgInnerMDPS ? kPixelPSZpitch : kStrip2SZpitch;
    float zGeom1 = alpaka::math::copysign(acc, zGeom, zIn);
    float rtLo = rtIn * (1.f + (zOut - zIn - zGeom1) / (zIn + zGeom1 + dLum) / dzDrtScale) -
                 rtGeom1;  //slope correction only on the lower end

    //Cut #1: rt condition
    float zInForHi = zIn - zGeom1 - dLum;
    if (zInForHi * zIn < 0) {
      zInForHi = alpaka::math::copysign(acc, 0.1f, zIn);
    }
    float rtHi = rtIn * (1.f + (zOut - zIn + zGeom1) / zInForHi) + rtGeom1;

    //Cut #2: rt condition
    if ((rtOut < rtLo) || (rtOut > rtHi))
      return false;

    float rIn = alpaka::math::sqrt(acc, zIn * zIn + rtIn * rtIn);

    const float drtSDIn = rtMid - rtIn;
    const float dzSDIn = zMid - zIn;
    const float dr3SDIn =
        alpaka::math::sqrt(acc, rtMid * rtMid + zMid * zMid) - alpaka::math::sqrt(acc, rtIn * rtIn + zIn * zIn);

    const float coshEta = dr3SDIn / drtSDIn;  //direction estimate
    const float dzOutInAbs = alpaka::math::abs(acc, zOut - zIn);
    const float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);
    const float zGeom1_another = kPixelPSZpitch;
    const float kZ = (zOut - zIn) / dzSDIn;
    float drtErr =
        zGeom1_another * zGeom1_another * drtSDIn * drtSDIn / dzSDIn / dzSDIn * (1.f - 2.f * kZ + 2.f * kZ * kZ);
    const float thetaMuls2 = (kMulsInGeV * kMulsInGeV) * (0.1f + 0.2 * (rtOut - rtIn) / 50.f) * (rIn / rtIn);
    const float muls2 = thetaMuls2 * 9.f / (ptCut * ptCut) * 16.f;
    drtErr += muls2 * multDzDr * multDzDr / 3.f * coshEta * coshEta;
    drtErr = alpaka::math::sqrt(acc, drtErr);

    //Cut #3: rt-z pointed

    if ((kZ < 0) || (rtOut < rtLo) || (rtOut > rtHi))
      return false;

    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InOut = mdsInGPU.anchorRt[secondMDIndex];

    float sdIn_alpha = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);

    float tl_axis_x = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float tl_axis_y = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];

    betaIn = sdIn_alpha - phi_mpi_pi(acc, phi(acc, tl_axis_x, tl_axis_y) - mdsInGPU.anchorPhi[firstMDIndex]);

    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;

    float swapTemp;

    if (alpaka::math::abs(acc, betaInRHmin) > alpaka::math::abs(acc, betaInRHmax)) {
      swapTemp = betaInRHmin;
      betaInRHmin = betaInRHmax;
      betaInRHmax = swapTemp;
    }

    float sdIn_dr = alpaka::math::sqrt(acc,
                                       (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) *
                                               (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) +
                                           (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]) *
                                               (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]));
    float sdIn_d = rt_InOut - rt_InLo;

    float dr = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
    betaInCut = alpaka::math::asin(acc, alpaka::math::min(acc, (-sdIn_dr + dr) * k2Rinv1GeVf / ptCut, kSinAlphaMax)) +
                (0.02f / sdIn_d);

    //Cut #4: first beta cut
    return alpaka::math::abs(acc, betaInRHmin) < betaInCut;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPointingConstraintEEE(TAcc const& acc,
                                                                Modules const& modulesInGPU,
                                                                MiniDoublets const& mdsInGPU,
                                                                Segments const& segmentsInGPU,
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
                                                                float& betaInCut) {
    float rtIn = mdsInGPU.anchorRt[firstMDIndex];
    float rtMid = mdsInGPU.anchorRt[secondMDIndex];
    rtOut = mdsInGPU.anchorRt[thirdMDIndex];

    float zIn = mdsInGPU.anchorZ[firstMDIndex];
    float zMid = mdsInGPU.anchorZ[secondMDIndex];
    zOut = mdsInGPU.anchorZ[thirdMDIndex];

    float alpha1GeV_Out = alpaka::math::asin(acc, alpaka::math::min(acc, rtOut * k2Rinv1GeVf / ptCut, kSinAlphaMax));

    float dzDrtScale =
        alpaka::math::tan(acc, alpha1GeV_Out) / alpha1GeV_Out;  // The track can bend in r-z plane slightly

    // Cut #0: Preliminary (Only here in endcap case)
    if (zIn * zOut <= 0)
      return false;

    float dLum = alpaka::math::copysign(acc, kDeltaZLum, zIn);
    bool isOutSgOuterMDPS = modulesInGPU.moduleType[outerOuterLowerModuleIndex] == PS;
    bool isInSgInnerMDPS = modulesInGPU.moduleType[innerInnerLowerModuleIndex] == PS;

    float rtGeom = (isInSgInnerMDPS and isOutSgOuterMDPS)  ? 2.f * kPixelPSZpitch
                   : (isInSgInnerMDPS or isOutSgOuterMDPS) ? kPixelPSZpitch + kStrip2SZpitch
                                                           : 2.f * kStrip2SZpitch;

    float dz = zOut - zIn;
    const float rtLo = rtIn * (1.f + dz / (zIn + dLum) / dzDrtScale) - rtGeom;  //slope correction only on the lower end
    const float rtHi = rtIn * (1.f + dz / (zIn - dLum)) + rtGeom;

    //Cut #1: rt condition
    if ((rtOut < rtLo) || (rtOut > rtHi))
      return false;

    bool isInSgOuterMDPS = modulesInGPU.moduleType[outerOuterLowerModuleIndex] == PS;

    float drtSDIn = rtMid - rtIn;
    float dzSDIn = zMid - zIn;
    float dr3SDIn =
        alpaka::math::sqrt(acc, rtMid * rtMid + zMid * zMid) - alpaka::math::sqrt(acc, rtIn * rtIn + zIn * zIn);

    float coshEta = dr3SDIn / drtSDIn;  //direction estimate
    float dzOutInAbs = alpaka::math::abs(acc, zOut - zIn);
    float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);

    float kZ = (zOut - zIn) / dzSDIn;
    float thetaMuls2 = (kMulsInGeV * kMulsInGeV) * (0.1f + 0.2f * (rtOut - rtIn) / 50.f);

    float muls2 = thetaMuls2 * 9.f / (ptCut * ptCut) * 16.f;

    float drtErr =
        alpaka::math::sqrt(acc,
                           kPixelPSZpitch * kPixelPSZpitch * 2.f / (dzSDIn * dzSDIn) * (dzOutInAbs * dzOutInAbs) +
                               muls2 * multDzDr * multDzDr / 3.f * coshEta * coshEta);

    float drtMean = drtSDIn * dzOutInAbs / alpaka::math::abs(acc, dzSDIn);
    float rtWindow = drtErr + rtGeom;
    float rtLo_point = rtIn + drtMean / dzDrtScale - rtWindow;
    float rtHi_point = rtIn + drtMean + rtWindow;

    // Cut #3: rt-z pointed
    // https://github.com/slava77/cms-tkph2-ntuple/blob/superDoubletLinked-91X-noMock/doubletAnalysis.C#L3765

    if (isInSgInnerMDPS and isInSgOuterMDPS)  // If both PS then we can point
    {
      if ((kZ < 0) || (rtOut < rtLo_point) || (rtOut > rtHi_point))
        return false;
    }

    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InOut = mdsInGPU.anchorRt[secondMDIndex];
    float sdIn_alpha = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);

    float tl_axis_x = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float tl_axis_y = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];

    betaIn = sdIn_alpha - phi_mpi_pi(acc, phi(acc, tl_axis_x, tl_axis_y) - mdsInGPU.anchorPhi[firstMDIndex]);

    float sdIn_alphaRHmin = __H2F(segmentsInGPU.dPhiChangeMins[innerSegmentIndex]);
    float sdIn_alphaRHmax = __H2F(segmentsInGPU.dPhiChangeMaxs[innerSegmentIndex]);
    float betaInRHmin = betaIn + sdIn_alphaRHmin - sdIn_alpha;
    float betaInRHmax = betaIn + sdIn_alphaRHmax - sdIn_alpha;

    float swapTemp;

    if (alpaka::math::abs(acc, betaInRHmin) > alpaka::math::abs(acc, betaInRHmax)) {
      swapTemp = betaInRHmin;
      betaInRHmin = betaInRHmax;
      betaInRHmax = swapTemp;
    }
    float sdIn_dr = alpaka::math::sqrt(acc,
                                       (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) *
                                               (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) +
                                           (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]) *
                                               (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]));
    float sdIn_d = rt_InOut - rt_InLo;

    float dr = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
    betaInCut = alpaka::math::asin(acc, alpaka::math::min(acc, (-sdIn_dr + dr) * k2Rinv1GeVf / ptCut, kSinAlphaMax)) +
                (0.02f / sdIn_d);

    //Cut #4: first beta cut
    return alpaka::math::abs(acc, betaInRHmin) < betaInCut;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPointingConstraint(TAcc const& acc,
                                                             Modules const& modulesInGPU,
                                                             MiniDoublets const& mdsInGPU,
                                                             Segments const& segmentsInGPU,
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
                                                             float& betaInCut) {
    short innerInnerLowerModuleSubdet = modulesInGPU.subdets[innerInnerLowerModuleIndex];
    short middleLowerModuleSubdet = modulesInGPU.subdets[middleLowerModuleIndex];
    short outerOuterLowerModuleSubdet = modulesInGPU.subdets[outerOuterLowerModuleIndex];

    if (innerInnerLowerModuleSubdet == Barrel and middleLowerModuleSubdet == Barrel and
        outerOuterLowerModuleSubdet == Barrel) {
      return passPointingConstraintBBB(acc,
                                       modulesInGPU,
                                       mdsInGPU,
                                       segmentsInGPU,
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
                                       betaInCut);
    } else if (innerInnerLowerModuleSubdet == Barrel and middleLowerModuleSubdet == Barrel and
               outerOuterLowerModuleSubdet == Endcap) {
      return passPointingConstraintBBE(acc,
                                       modulesInGPU,
                                       mdsInGPU,
                                       segmentsInGPU,
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
                                       betaInCut);
    } else if (innerInnerLowerModuleSubdet == Barrel and middleLowerModuleSubdet == Endcap and
               outerOuterLowerModuleSubdet == Endcap) {
      return passPointingConstraintBBE(acc,
                                       modulesInGPU,
                                       mdsInGPU,
                                       segmentsInGPU,
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
                                       betaInCut);

    }

    else if (innerInnerLowerModuleSubdet == Endcap and middleLowerModuleSubdet == Endcap and
             outerOuterLowerModuleSubdet == Endcap) {
      return passPointingConstraintEEE(acc,
                                       modulesInGPU,
                                       mdsInGPU,
                                       segmentsInGPU,
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
                                       betaInCut);
    }
    return false;  // failsafe
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computeRadiusFromThreeAnchorHits(
      TAcc const& acc, float x1, float y1, float x2, float y2, float x3, float y3, float& g, float& f) {
    float radius = 0.f;

    //(g,f) -> center
    //first anchor hit - (x1,y1), second anchor hit - (x2,y2), third anchor hit - (x3, y3)

    float denomInv = 1.0f / ((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3));

    float xy1sqr = x1 * x1 + y1 * y1;

    float xy2sqr = x2 * x2 + y2 * y2;

    float xy3sqr = x3 * x3 + y3 * y3;

    g = 0.5f * ((y3 - y2) * xy1sqr + (y1 - y3) * xy2sqr + (y2 - y1) * xy3sqr) * denomInv;

    f = 0.5f * ((x2 - x3) * xy1sqr + (x3 - x1) * xy2sqr + (x1 - x2) * xy3sqr) * denomInv;

    float c = ((x2 * y3 - x3 * y2) * xy1sqr + (x3 * y1 - x1 * y3) * xy2sqr + (x1 * y2 - x2 * y1) * xy3sqr) * denomInv;

    if (((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3) == 0) || (g * g + f * f - c < 0)) {
#ifdef WARNINGS
      printf("three collinear points or FATAL! r^2 < 0!\n");
#endif
      radius = -1.f;
    } else
      radius = alpaka::math::sqrt(acc, g * g + f * f - c);

    return radius;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runTripletConstraintsAndAlgo(TAcc const& acc,
                                                                   Modules const& modulesInGPU,
                                                                   MiniDoublets const& mdsInGPU,
                                                                   Segments const& segmentsInGPU,
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
                                                                   float& circleCenterY) {
    //this cut reduces the number of candidates by a factor of 4, i.e., 3 out of 4 warps can end right here!
    if (segmentsInGPU.mdIndices[2 * innerSegmentIndex + 1] != segmentsInGPU.mdIndices[2 * outerSegmentIndex])
      return false;

    unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex];
    unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex];
    unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex + 1];

    if (not(passRZConstraint(acc,
                             modulesInGPU,
                             mdsInGPU,
                             segmentsInGPU,
                             innerInnerLowerModuleIndex,
                             middleLowerModuleIndex,
                             outerOuterLowerModuleIndex,
                             firstMDIndex,
                             secondMDIndex,
                             thirdMDIndex)))
      return false;
    if (not(passPointingConstraint(acc,
                                   modulesInGPU,
                                   mdsInGPU,
                                   segmentsInGPU,
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
                                   betaInCut)))
      return false;

    float x1 = mdsInGPU.anchorX[firstMDIndex];
    float x2 = mdsInGPU.anchorX[secondMDIndex];
    float x3 = mdsInGPU.anchorX[thirdMDIndex];
    float y1 = mdsInGPU.anchorY[firstMDIndex];
    float y2 = mdsInGPU.anchorY[secondMDIndex];
    float y3 = mdsInGPU.anchorY[thirdMDIndex];

    circleRadius = computeRadiusFromThreeAnchorHits(acc, x1, y1, x2, y2, x3, y3, circleCenterX, circleCenterY);
    return true;
  }

  struct CreateTripletsInGPUv2 {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  Modules modulesInGPU,
                                  MiniDoublets mdsInGPU,
                                  Segments segmentsInGPU,
                                  Triplets tripletsInGPU,
                                  ObjectRanges rangesInGPU,
                                  uint16_t* index_gpu,
                                  uint16_t nonZeroModules) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (uint16_t innerLowerModuleArrayIdx = globalThreadIdx[0]; innerLowerModuleArrayIdx < nonZeroModules;
           innerLowerModuleArrayIdx += gridThreadExtent[0]) {
        uint16_t innerInnerLowerModuleIndex = index_gpu[innerLowerModuleArrayIdx];
        if (innerInnerLowerModuleIndex >= *modulesInGPU.nLowerModules)
          continue;

        uint16_t nConnectedModules = modulesInGPU.nConnectedModules[innerInnerLowerModuleIndex];
        if (nConnectedModules == 0)
          continue;

        unsigned int nInnerSegments = segmentsInGPU.nSegments[innerInnerLowerModuleIndex];
        for (unsigned int innerSegmentArrayIndex = globalThreadIdx[1]; innerSegmentArrayIndex < nInnerSegments;
             innerSegmentArrayIndex += gridThreadExtent[1]) {
          unsigned int innerSegmentIndex =
              rangesInGPU.segmentRanges[innerInnerLowerModuleIndex * 2] + innerSegmentArrayIndex;

          // middle lower module - outer lower module of inner segment
          uint16_t middleLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[innerSegmentIndex];

          unsigned int nOuterSegments = segmentsInGPU.nSegments[middleLowerModuleIndex];
          for (unsigned int outerSegmentArrayIndex = globalThreadIdx[2]; outerSegmentArrayIndex < nOuterSegments;
               outerSegmentArrayIndex += gridThreadExtent[2]) {
            unsigned int outerSegmentIndex =
                rangesInGPU.segmentRanges[2 * middleLowerModuleIndex] + outerSegmentArrayIndex;

            uint16_t outerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[outerSegmentIndex];

            float zOut, rtOut, betaIn, betaInCut, circleRadius, circleCenterX, circleCenterY;

            bool success = runTripletConstraintsAndAlgo(acc,
                                                        modulesInGPU,
                                                        mdsInGPU,
                                                        segmentsInGPU,
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
                                                        circleCenterY);

            if (success) {
              unsigned int totOccupancyTriplets =
                  alpaka::atomicAdd(acc,
                                    &tripletsInGPU.totOccupancyTriplets[innerInnerLowerModuleIndex],
                                    1u,
                                    alpaka::hierarchy::Threads{});
              if (static_cast<int>(totOccupancyTriplets) >=
                  rangesInGPU.tripletModuleOccupancy[innerInnerLowerModuleIndex]) {
#ifdef WARNINGS
                printf("Triplet excess alert! Module index = %d\n", innerInnerLowerModuleIndex);
#endif
              } else {
                unsigned int tripletModuleIndex = alpaka::atomicAdd(
                    acc, &tripletsInGPU.nTriplets[innerInnerLowerModuleIndex], 1u, alpaka::hierarchy::Threads{});
                unsigned int tripletIndex =
                    rangesInGPU.tripletModuleIndices[innerInnerLowerModuleIndex] + tripletModuleIndex;
#ifdef CUT_VALUE_DEBUG
                addTripletToMemory(modulesInGPU,
                                   mdsInGPU,
                                   segmentsInGPU,
                                   tripletsInGPU,
                                   innerSegmentIndex,
                                   outerSegmentIndex,
                                   innerInnerLowerModuleIndex,
                                   middleLowerModuleIndex,
                                   outerOuterLowerModuleIndex,
                                   zOut,
                                   rtOut,
                                   betaIn,
                                   betaInCut,
                                   circleRadius,
                                   circleCenterX,
                                   circleCenterY,
                                   tripletIndex);
#else
                addTripletToMemory(modulesInGPU,
                                   mdsInGPU,
                                   segmentsInGPU,
                                   tripletsInGPU,
                                   innerSegmentIndex,
                                   outerSegmentIndex,
                                   innerInnerLowerModuleIndex,
                                   middleLowerModuleIndex,
                                   outerOuterLowerModuleIndex,
                                   betaIn,
                                   circleRadius,
                                   circleCenterX,
                                   circleCenterY,
                                   tripletIndex);
#endif
              }
            }
          }
        }
      }
    }
  };

  struct CreateTripletArrayRanges {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  Modules modulesInGPU,
                                  ObjectRanges rangesInGPU,
                                  Segments segmentsInGPU) const {
      // implementation is 1D with a single block
      static_assert(std::is_same_v<TAcc, ALPAKA_ACCELERATOR_NAMESPACE::Acc1D>, "Should be Acc1D");
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      // Initialize variables in shared memory and set to 0
      int& nTotalTriplets = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        nTotalTriplets = 0;
      }
      alpaka::syncBlockThreads(acc);

      // Create variables outside of the for loop.
      int occupancy, category_number, eta_number;

      for (uint16_t i = globalThreadIdx[0]; i < *modulesInGPU.nLowerModules; i += gridThreadExtent[0]) {
        if (segmentsInGPU.nSegments[i] == 0) {
          rangesInGPU.tripletModuleIndices[i] = nTotalTriplets;
          rangesInGPU.tripletModuleOccupancy[i] = 0;
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
          occupancy = 543;
        else if (category_number == 0 && eta_number == 1)
          occupancy = 235;
        else if (category_number == 0 && eta_number == 2)
          occupancy = 88;
        else if (category_number == 0 && eta_number == 3)
          occupancy = 46;
        else if (category_number == 1 && eta_number == 0)
          occupancy = 755;
        else if (category_number == 1 && eta_number == 1)
          occupancy = 347;
        else if (category_number == 2 && eta_number == 1)
          occupancy = 0;
        else if (category_number == 2 && eta_number == 2)
          occupancy = 0;
        else if (category_number == 3 && eta_number == 1)
          occupancy = 38;
        else if (category_number == 3 && eta_number == 2)
          occupancy = 46;
        else if (category_number == 3 && eta_number == 3)
          occupancy = 39;
        else {
          occupancy = 0;
#ifdef WARNINGS
          printf("Unhandled case in createTripletArrayRanges! Module index = %i\n", i);
#endif
        }

        rangesInGPU.tripletModuleOccupancy[i] = occupancy;
        unsigned int nTotT = alpaka::atomicAdd(acc, &nTotalTriplets, occupancy, alpaka::hierarchy::Threads{});
        rangesInGPU.tripletModuleIndices[i] = nTotT;
      }

      // Wait for all threads to finish before reporting final values
      alpaka::syncBlockThreads(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        *rangesInGPU.device_nTotalTrips = nTotalTriplets;
      }
    }
  };

  struct AddTripletRangesToEventExplicit {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  Modules modulesInGPU,
                                  Triplets tripletsInGPU,
                                  ObjectRanges rangesInGPU) const {
      // implementation is 1D with a single block
      static_assert(std::is_same_v<TAcc, ALPAKA_ACCELERATOR_NAMESPACE::Acc1D>, "Should be Acc1D");
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (uint16_t i = globalThreadIdx[0]; i < *modulesInGPU.nLowerModules; i += gridThreadExtent[0]) {
        if (tripletsInGPU.nTriplets[i] == 0) {
          rangesInGPU.tripletRanges[i * 2] = -1;
          rangesInGPU.tripletRanges[i * 2 + 1] = -1;
        } else {
          rangesInGPU.tripletRanges[i * 2] = rangesInGPU.tripletModuleIndices[i];
          rangesInGPU.tripletRanges[i * 2 + 1] = rangesInGPU.tripletModuleIndices[i] + tripletsInGPU.nTriplets[i] - 1;
        }
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
