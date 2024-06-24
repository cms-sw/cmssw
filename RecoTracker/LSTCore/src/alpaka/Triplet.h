#ifndef Triplet_cuh
#define Triplet_cuh

#ifdef LST_IS_CMSSW_PACKAGE
#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/alpaka/Module.h"
#else
#include "Constants.h"
#include "Module.h"
#endif

#include "Segment.h"
#include "MiniDoublet.h"
#include "Hit.h"

namespace SDL {
  struct triplets {
    unsigned int* segmentIndices;
    uint16_t* lowerModuleIndices;  //3 of them now
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
    float* deltaPhiPos;
    float* deltaPhi;
    float* zLo;
    float* zHi;
    float* zLoPointed;
    float* zHiPointed;
    float* sdlCut;
    float* betaInCut;
    float* rtLo;
    float* rtHi;
#endif
    template <typename TBuff>
    void setData(TBuff& tripletsbuf) {
      segmentIndices = alpaka::getPtrNative(tripletsbuf.segmentIndices_buf);
      lowerModuleIndices = alpaka::getPtrNative(tripletsbuf.lowerModuleIndices_buf);
      nTriplets = alpaka::getPtrNative(tripletsbuf.nTriplets_buf);
      totOccupancyTriplets = alpaka::getPtrNative(tripletsbuf.totOccupancyTriplets_buf);
      nMemoryLocations = alpaka::getPtrNative(tripletsbuf.nMemoryLocations_buf);
      logicalLayers = alpaka::getPtrNative(tripletsbuf.logicalLayers_buf);
      hitIndices = alpaka::getPtrNative(tripletsbuf.hitIndices_buf);
      betaIn = alpaka::getPtrNative(tripletsbuf.betaIn_buf);
      circleRadius = alpaka::getPtrNative(tripletsbuf.circleRadius_buf);
      circleCenterX = alpaka::getPtrNative(tripletsbuf.circleCenterX_buf);
      circleCenterY = alpaka::getPtrNative(tripletsbuf.circleCenterY_buf);
      partOfPT5 = alpaka::getPtrNative(tripletsbuf.partOfPT5_buf);
      partOfT5 = alpaka::getPtrNative(tripletsbuf.partOfT5_buf);
      partOfPT3 = alpaka::getPtrNative(tripletsbuf.partOfPT3_buf);
#ifdef CUT_VALUE_DEBUG
      zOut = alpaka::getPtrNative(tripletsbuf.zOut_buf);
      rtOut = alpaka::getPtrNative(tripletsbuf.rtOut_buf);
      deltaPhiPos = alpaka::getPtrNative(tripletsbuf.deltaPhiPos_buf);
      deltaPhi = alpaka::getPtrNative(tripletsbuf.deltaPhi_buf);
      zLo = alpaka::getPtrNative(tripletsbuf.zLo_buf);
      zHi = alpaka::getPtrNative(tripletsbuf.zHi_buf);
      zLoPointed = alpaka::getPtrNative(tripletsbuf.zLoPointed_buf);
      zHiPointed = alpaka::getPtrNative(tripletsbuf.zHiPointed_buf);
      sdlCut = alpaka::getPtrNative(tripletsbuf.sdlCut_buf);
      betaInCut = alpaka::getPtrNative(tripletsbuf.betaInCut_buf);
      rtLo = alpaka::getPtrNative(tripletsbuf.rtLo_buf);
      rtHi = alpaka::getPtrNative(tripletsbuf.rtHi_buf);
#endif
    }
  };

  template <typename TDev>
  struct tripletsBuffer : triplets {
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
    Buf<TDev, float> sdlCut_buf;
    Buf<TDev, float> betaInCut_buf;
    Buf<TDev, float> rtLo_buf;
    Buf<TDev, float> rtHi_buf;
#endif

    template <typename TQueue, typename TDevAcc>
    tripletsBuffer(unsigned int maxTriplets, unsigned int nLowerModules, TDevAcc const& devAccIn, TQueue& queue)
        : segmentIndices_buf(allocBufWrapper<unsigned int>(devAccIn, 2 * maxTriplets, queue)),
          lowerModuleIndices_buf(allocBufWrapper<uint16_t>(devAccIn, 3 * maxTriplets, queue)),
          nTriplets_buf(allocBufWrapper<unsigned int>(devAccIn, nLowerModules, queue)),
          totOccupancyTriplets_buf(allocBufWrapper<unsigned int>(devAccIn, nLowerModules, queue)),
          nMemoryLocations_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          logicalLayers_buf(allocBufWrapper<uint8_t>(devAccIn, maxTriplets * 3, queue)),
          hitIndices_buf(allocBufWrapper<unsigned int>(devAccIn, maxTriplets * 6, queue)),
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
          sdlCut_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
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
  };

#ifdef CUT_VALUE_DEBUG
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addTripletToMemory(struct SDL::modules& modulesInGPU,
                                                         struct SDL::miniDoublets& mdsInGPU,
                                                         struct SDL::segments& segmentsInGPU,
                                                         struct SDL::triplets& tripletsInGPU,
                                                         unsigned int& innerSegmentIndex,
                                                         unsigned int& outerSegmentIndex,
                                                         uint16_t& innerInnerLowerModuleIndex,
                                                         uint16_t& middleLowerModuleIndex,
                                                         uint16_t& outerOuterLowerModuleIndex,
                                                         float& zOut,
                                                         float& rtOut,
                                                         float& deltaPhiPos,
                                                         float& deltaPhi,
                                                         float& betaIn,
                                                         float& circleRadius,
                                                         float& circleCenterX,
                                                         float& circleCenterY,
                                                         float& zLo,
                                                         float& zHi,
                                                         float& rtLo,
                                                         float& rtHi,
                                                         float& zLoPointed,
                                                         float& zHiPointed,
                                                         float& sdlCut,
                                                         float& betaInCut,
                                                         unsigned int& tripletIndex)
#else
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addTripletToMemory(struct SDL::modules& modulesInGPU,
                                                         struct SDL::miniDoublets& mdsInGPU,
                                                         struct SDL::segments& segmentsInGPU,
                                                         struct SDL::triplets& tripletsInGPU,
                                                         unsigned int& innerSegmentIndex,
                                                         unsigned int& outerSegmentIndex,
                                                         uint16_t& innerInnerLowerModuleIndex,
                                                         uint16_t& middleLowerModuleIndex,
                                                         uint16_t& outerOuterLowerModuleIndex,
                                                         float& betaIn,
                                                         float& circleRadius,
                                                         float& circleCenterX,
                                                         float& circleCenterY,
                                                         unsigned int& tripletIndex)
#endif
  {
    tripletsInGPU.segmentIndices[tripletIndex * 2] = innerSegmentIndex;
    tripletsInGPU.segmentIndices[tripletIndex * 2 + 1] = outerSegmentIndex;
    tripletsInGPU.lowerModuleIndices[tripletIndex * 3] = innerInnerLowerModuleIndex;
    tripletsInGPU.lowerModuleIndices[tripletIndex * 3 + 1] = middleLowerModuleIndex;
    tripletsInGPU.lowerModuleIndices[tripletIndex * 3 + 2] = outerOuterLowerModuleIndex;

    tripletsInGPU.betaIn[tripletIndex] = __F2H(betaIn);
    tripletsInGPU.circleRadius[tripletIndex] = circleRadius;
    tripletsInGPU.circleCenterX[tripletIndex] = circleCenterX;
    tripletsInGPU.circleCenterY[tripletIndex] = circleCenterY;
    tripletsInGPU.logicalLayers[tripletIndex * 3] =
        modulesInGPU.layers[innerInnerLowerModuleIndex] + (modulesInGPU.subdets[innerInnerLowerModuleIndex] == 4) * 6;
    tripletsInGPU.logicalLayers[tripletIndex * 3 + 1] =
        modulesInGPU.layers[middleLowerModuleIndex] + (modulesInGPU.subdets[middleLowerModuleIndex] == 4) * 6;
    tripletsInGPU.logicalLayers[tripletIndex * 3 + 2] =
        modulesInGPU.layers[outerOuterLowerModuleIndex] + (modulesInGPU.subdets[outerOuterLowerModuleIndex] == 4) * 6;
    //get the hits
    unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex];
    unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex + 1];
    unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex + 1];

    tripletsInGPU.hitIndices[tripletIndex * 6] = mdsInGPU.anchorHitIndices[firstMDIndex];
    tripletsInGPU.hitIndices[tripletIndex * 6 + 1] = mdsInGPU.outerHitIndices[firstMDIndex];
    tripletsInGPU.hitIndices[tripletIndex * 6 + 2] = mdsInGPU.anchorHitIndices[secondMDIndex];
    tripletsInGPU.hitIndices[tripletIndex * 6 + 3] = mdsInGPU.outerHitIndices[secondMDIndex];
    tripletsInGPU.hitIndices[tripletIndex * 6 + 4] = mdsInGPU.anchorHitIndices[thirdMDIndex];
    tripletsInGPU.hitIndices[tripletIndex * 6 + 5] = mdsInGPU.outerHitIndices[thirdMDIndex];
#ifdef CUT_VALUE_DEBUG
    tripletsInGPU.zOut[tripletIndex] = zOut;
    tripletsInGPU.rtOut[tripletIndex] = rtOut;
    tripletsInGPU.deltaPhiPos[tripletIndex] = deltaPhiPos;
    tripletsInGPU.deltaPhi[tripletIndex] = deltaPhi;
    tripletsInGPU.zLo[tripletIndex] = zLo;
    tripletsInGPU.zHi[tripletIndex] = zHi;
    tripletsInGPU.rtLo[tripletIndex] = rtLo;
    tripletsInGPU.rtHi[tripletIndex] = rtHi;
    tripletsInGPU.zLoPointed[tripletIndex] = zLoPointed;
    tripletsInGPU.zHiPointed[tripletIndex] = zHiPointed;
    tripletsInGPU.sdlCut[tripletIndex] = sdlCut;
    tripletsInGPU.betaInCut[tripletIndex] = betaInCut;
#endif
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passRZConstraint(TAcc const& acc,
                                                       struct SDL::modules& modulesInGPU,
                                                       struct SDL::miniDoublets& mdsInGPU,
                                                       struct SDL::segments& segmentsInGPU,
                                                       uint16_t& innerInnerLowerModuleIndex,
                                                       uint16_t& middleLowerModuleIndex,
                                                       uint16_t& outerOuterLowerModuleIndex,
                                                       unsigned int& firstMDIndex,
                                                       unsigned int& secondMDIndex,
                                                       unsigned int& thirdMDIndex) {
    //get the rt and z
    const float& r1 = mdsInGPU.anchorRt[firstMDIndex];
    const float& r2 = mdsInGPU.anchorRt[secondMDIndex];
    const float& r3 = mdsInGPU.anchorRt[thirdMDIndex];

    const float& z1 = mdsInGPU.anchorZ[firstMDIndex];
    const float& z2 = mdsInGPU.anchorZ[secondMDIndex];
    const float& z3 = mdsInGPU.anchorZ[thirdMDIndex];

    //following Philip's layer number prescription
    const int layer1 = modulesInGPU.sdlLayers[innerInnerLowerModuleIndex];
    const int layer2 = modulesInGPU.sdlLayers[middleLowerModuleIndex];
    const int layer3 = modulesInGPU.sdlLayers[outerOuterLowerModuleIndex];

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
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPointingConstraintBBB(TAcc const& acc,
                                                                struct SDL::modules& modulesInGPU,
                                                                struct SDL::miniDoublets& mdsInGPU,
                                                                struct SDL::segments& segmentsInGPU,
                                                                uint16_t& innerInnerLowerModuleIndex,
                                                                uint16_t& middleLowerModuleIndex,
                                                                uint16_t& outerOuterLowerModuleIndex,
                                                                unsigned int& firstMDIndex,
                                                                unsigned int& secondMDIndex,
                                                                unsigned int& thirdMDIndex,
                                                                float& zOut,
                                                                float& rtOut,
                                                                unsigned int& innerSegmentIndex,
                                                                float& betaIn,
                                                                float& betaInCut) {
    bool pass = true;
    bool isPSIn = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS);
    bool isPSOut = (modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::PS);

    float rtIn = mdsInGPU.anchorRt[firstMDIndex];
    float rtMid = mdsInGPU.anchorRt[secondMDIndex];
    rtOut = mdsInGPU.anchorRt[thirdMDIndex];

    float zIn = mdsInGPU.anchorZ[firstMDIndex];
    float zMid = mdsInGPU.anchorZ[secondMDIndex];
    zOut = mdsInGPU.anchorZ[thirdMDIndex];

    float alpha1GeVOut =
        alpaka::math::asin(acc, alpaka::math::min(acc, rtOut * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax));

    float rtRatio_OutIn = rtOut / rtIn;  // Outer segment beginning rt divided by inner segment beginning rt;
    float dzDrtScale = alpaka::math::tan(acc, alpha1GeVOut) / alpha1GeVOut;  // The track can bend in r-z plane slightly
    float zpitchIn = (isPSIn ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
    float zpitchOut = (isPSOut ? SDL::pixelPSZpitch : SDL::strip2SZpitch);

    const float zHi =
        zIn + (zIn + SDL::deltaZLum) * (rtRatio_OutIn - 1.f) * (zIn < 0.f ? 1.f : dzDrtScale) + (zpitchIn + zpitchOut);
    const float zLo = zIn + (zIn - SDL::deltaZLum) * (rtRatio_OutIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) -
                      (zpitchIn + zpitchOut);  //slope-correction only on outer end

    //Cut 1 - z compatibility
    pass = pass and ((zOut >= zLo) && (zOut <= zHi));
    if (not pass)
      return pass;

    float drt_OutIn = (rtOut - rtIn);

    float r3In = alpaka::math::sqrt(acc, zIn * zIn + rtIn * rtIn);
    float drt_InSeg = rtMid - rtIn;
    float dz_InSeg = zMid - zIn;
    float dr3_InSeg =
        alpaka::math::sqrt(acc, rtMid * rtMid + zMid * zMid) - alpaka::math::sqrt(acc, rtIn * rtIn + zIn + zIn);

    float coshEta = dr3_InSeg / drt_InSeg;
    float dzErr = (zpitchIn + zpitchOut) * (zpitchIn + zpitchOut) * 2.f;

    float sdlThetaMulsF =
        0.015f * alpaka::math::sqrt(acc, 0.1f + 0.2f * (rtOut - rtIn) / 50.f) * alpaka::math::sqrt(acc, r3In / rtIn);
    float sdlMuls = sdlThetaMulsF * 3.f / SDL::ptCut * 4.f;                        // will need a better guess than x4?
    dzErr += sdlMuls * sdlMuls * drt_OutIn * drt_OutIn / 3.f * coshEta * coshEta;  //sloppy
    dzErr = alpaka::math::sqrt(acc, dzErr);

    // Constructing upper and lower bound
    const float dzMean = dz_InSeg / drt_InSeg * drt_OutIn;
    const float zWindow =
        dzErr / drt_InSeg * drt_OutIn +
        (zpitchIn + zpitchOut);  //FIXME for SDL::ptCut lower than ~0.8 need to add curv path correction
    const float zLoPointed = zIn + dzMean * (zIn > 0.f ? 1.f : dzDrtScale) - zWindow;
    const float zHiPointed = zIn + dzMean * (zIn < 0.f ? 1.f : dzDrtScale) + zWindow;

    // Constructing upper and lower bound

    // Cut #2: Pointed Z (Inner segment two MD points to outer segment inner MD)
    pass = pass and ((zOut >= zLoPointed) && (zOut <= zHiPointed));

    // raw betaIn value without any correction, based on the mini-doublet hit positions
    float alpha_InLo = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
    float tl_axis_x = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float tl_axis_y = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];
    betaIn = alpha_InLo - SDL::phi_mpi_pi(acc, SDL::phi(acc, tl_axis_x, tl_axis_y) - mdsInGPU.anchorPhi[firstMDIndex]);

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
        alpaka::math::asin(
            acc, alpaka::math::min(acc, (-rt_InSeg + drt_tl_axis) * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) +
        (0.02f / drt_InSeg);

    //Cut #3: first beta cut
    pass = pass and (alpaka::math::abs(acc, betaIn) < betaInCut);

    return pass;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPointingConstraintBBE(TAcc const& acc,
                                                                struct SDL::modules& modulesInGPU,
                                                                struct SDL::miniDoublets& mdsInGPU,
                                                                struct SDL::segments& segmentsInGPU,
                                                                uint16_t& innerInnerLowerModuleIndex,
                                                                uint16_t& middleLowerModuleIndex,
                                                                uint16_t& outerOuterLowerModuleIndex,
                                                                unsigned int& firstMDIndex,
                                                                unsigned int& secondMDIndex,
                                                                unsigned int& thirdMDIndex,
                                                                float& zOut,
                                                                float& rtOut,
                                                                uint16_t& innerOuterLowerModuleIndex,
                                                                unsigned int& innerSegmentIndex,
                                                                unsigned int& outerSegmentIndex,
                                                                float& betaIn,
                                                                float& betaInCut) {
    bool pass = true;

    bool isPSIn = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS);
    bool isPSOut = (modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::PS);

    float rtIn = mdsInGPU.anchorRt[firstMDIndex];
    float rtMid = mdsInGPU.anchorRt[secondMDIndex];
    rtOut = mdsInGPU.anchorRt[thirdMDIndex];

    float zIn = mdsInGPU.anchorZ[firstMDIndex];
    float zMid = mdsInGPU.anchorZ[secondMDIndex];
    zOut = mdsInGPU.anchorZ[thirdMDIndex];

    float alpha1GeV_OutLo =
        alpaka::math::asin(acc, alpaka::math::min(acc, rtOut * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax));

    float dzDrtScale =
        alpaka::math::tan(acc, alpha1GeV_OutLo) / alpha1GeV_OutLo;  // The track can bend in r-z plane slightly
    float zpitchIn = (isPSIn ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
    float zpitchOut = (isPSOut ? SDL::pixelPSZpitch : SDL::strip2SZpitch);
    float zGeom = zpitchIn + zpitchOut;

    // Cut #0: Preliminary (Only here in endcap case)
    pass = pass and (zIn * zOut > 0);
    if (not pass)
      return pass;

    float dLum = SDL::copysignf(SDL::deltaZLum, zIn);
    bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::PS;
    float rtGeom1 = isOutSgInnerMDPS ? SDL::pixelPSZpitch : SDL::strip2SZpitch;
    float zGeom1 = SDL::copysignf(zGeom, zIn);
    float rtLo = rtIn * (1.f + (zOut - zIn - zGeom1) / (zIn + zGeom1 + dLum) / dzDrtScale) -
                 rtGeom1;  //slope correction only on the lower end

    //Cut #1: rt condition
    float zInForHi = zIn - zGeom1 - dLum;
    if (zInForHi * zIn < 0) {
      zInForHi = SDL::copysignf(0.1f, zIn);
    }
    float rtHi = rtIn * (1.f + (zOut - zIn + zGeom1) / zInForHi) + rtGeom1;

    //Cut #2: rt condition
    pass = pass and ((rtOut >= rtLo) && (rtOut <= rtHi));
    if (not pass)
      return pass;

    float rIn = alpaka::math::sqrt(acc, zIn * zIn + rtIn * rtIn);

    const float drtSDIn = rtMid - rtIn;
    const float dzSDIn = zMid - zIn;
    const float dr3SDIn =
        alpaka::math::sqrt(acc, rtMid * rtMid + zMid * zMid) - alpaka::math::sqrt(acc, rtIn * rtIn + zIn * zIn);

    const float coshEta = dr3SDIn / drtSDIn;  //direction estimate
    const float dzOutInAbs = alpaka::math::abs(acc, zOut - zIn);
    const float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);
    const float zGeom1_another = SDL::pixelPSZpitch;
    const float kZ = (zOut - zIn) / dzSDIn;
    float drtErr =
        zGeom1_another * zGeom1_another * drtSDIn * drtSDIn / dzSDIn / dzSDIn * (1.f - 2.f * kZ + 2.f * kZ * kZ);
    const float sdlThetaMulsF =
        0.015f * alpaka::math::sqrt(acc, 0.1f + 0.2 * (rtOut - rtIn) / 50.f) * alpaka::math::sqrt(acc, rIn / rtIn);
    const float sdlMuls = sdlThetaMulsF * 3.f / SDL::ptCut * 4.f;  //will need a better guess than x4?
    drtErr +=
        sdlMuls * sdlMuls * multDzDr * multDzDr / 3.f * coshEta * coshEta;  //sloppy: relative muls is 1/3 of total muls
    drtErr = alpaka::math::sqrt(acc, drtErr);

    //Cut #3: rt-z pointed

    pass = pass and (kZ >= 0) && (rtOut >= rtLo) && (rtOut <= rtHi);

    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InOut = mdsInGPU.anchorRt[secondMDIndex];

    float sdIn_alpha = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);

    float tl_axis_x = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float tl_axis_y = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];

    betaIn = sdIn_alpha - SDL::phi_mpi_pi(acc, SDL::phi(acc, tl_axis_x, tl_axis_y) - mdsInGPU.anchorPhi[firstMDIndex]);

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
    betaInCut = alpaka::math::asin(
                    acc, alpaka::math::min(acc, (-sdIn_dr + dr) * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) +
                (0.02f / sdIn_d);

    //Cut #4: first beta cut
    pass = pass and (alpaka::math::abs(acc, betaInRHmin) < betaInCut);
    return pass;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPointingConstraintEEE(TAcc const& acc,
                                                                struct SDL::modules& modulesInGPU,
                                                                struct SDL::miniDoublets& mdsInGPU,
                                                                struct SDL::segments& segmentsInGPU,
                                                                uint16_t& innerInnerLowerModuleIndex,
                                                                uint16_t& middleLowerModuleIndex,
                                                                uint16_t& outerOuterLowerModuleIndex,
                                                                unsigned int& firstMDIndex,
                                                                unsigned int& secondMDIndex,
                                                                unsigned int& thirdMDIndex,
                                                                float& zOut,
                                                                float& rtOut,
                                                                unsigned int& innerSegmentIndex,
                                                                unsigned int& outerSegmentIndex,
                                                                float& betaIn,
                                                                float& betaInCut) {
    bool pass = true;

    float rtIn = mdsInGPU.anchorRt[firstMDIndex];
    float rtMid = mdsInGPU.anchorRt[secondMDIndex];
    rtOut = mdsInGPU.anchorRt[thirdMDIndex];

    float zIn = mdsInGPU.anchorZ[firstMDIndex];
    float zMid = mdsInGPU.anchorZ[secondMDIndex];
    zOut = mdsInGPU.anchorZ[thirdMDIndex];

    float alpha1GeV_Out =
        alpaka::math::asin(acc, alpaka::math::min(acc, rtOut * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax));

    float dzDrtScale =
        alpaka::math::tan(acc, alpha1GeV_Out) / alpha1GeV_Out;  // The track can bend in r-z plane slightly

    // Cut #0: Preliminary (Only here in endcap case)
    pass = pass and (zIn * zOut > 0);
    if (not pass)
      return pass;

    float dLum = SDL::copysignf(SDL::deltaZLum, zIn);
    bool isOutSgOuterMDPS = modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::PS;
    bool isInSgInnerMDPS = modulesInGPU.moduleType[innerInnerLowerModuleIndex] == SDL::PS;

    float rtGeom = (isInSgInnerMDPS and isOutSgOuterMDPS)  ? 2.f * SDL::pixelPSZpitch
                   : (isInSgInnerMDPS or isOutSgOuterMDPS) ? SDL::pixelPSZpitch + SDL::strip2SZpitch
                                                           : 2.f * SDL::strip2SZpitch;

    float dz = zOut - zIn;
    const float rtLo = rtIn * (1.f + dz / (zIn + dLum) / dzDrtScale) - rtGeom;  //slope correction only on the lower end
    const float rtHi = rtIn * (1.f + dz / (zIn - dLum)) + rtGeom;

    //Cut #1: rt condition
    pass = pass and ((rtOut >= rtLo) && (rtOut <= rtHi));
    if (not pass)
      return pass;

    bool isInSgOuterMDPS = modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::PS;

    float drtSDIn = rtMid - rtIn;
    float dzSDIn = zMid - zIn;
    float dr3SDIn =
        alpaka::math::sqrt(acc, rtMid * rtMid + zMid * zMid) - alpaka::math::sqrt(acc, rtIn * rtIn + zIn * zIn);

    float coshEta = dr3SDIn / drtSDIn;  //direction estimate
    float dzOutInAbs = alpaka::math::abs(acc, zOut - zIn);
    float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);

    float kZ = (zOut - zIn) / dzSDIn;
    float sdlThetaMulsF = 0.015f * alpaka::math::sqrt(acc, 0.1f + 0.2f * (rtOut - rtIn) / 50.f);

    float sdlMuls = sdlThetaMulsF * 3.f / SDL::ptCut * 4.f;  //will need a better guess than x4?

    float drtErr = alpaka::math::sqrt(
        acc,
        SDL::pixelPSZpitch * SDL::pixelPSZpitch * 2.f / (dzSDIn * dzSDIn) * (dzOutInAbs * dzOutInAbs) +
            sdlMuls * sdlMuls * multDzDr * multDzDr / 3.f * coshEta * coshEta);

    float drtMean = drtSDIn * dzOutInAbs / alpaka::math::abs(acc, dzSDIn);
    float rtWindow = drtErr + rtGeom;
    float rtLo_point = rtIn + drtMean / dzDrtScale - rtWindow;
    float rtHi_point = rtIn + drtMean + rtWindow;

    // Cut #3: rt-z pointed
    // https://github.com/slava77/cms-tkph2-ntuple/blob/superDoubletLinked-91X-noMock/doubletAnalysis.C#L3765

    if (isInSgInnerMDPS and isInSgOuterMDPS)  // If both PS then we can point
    {
      pass = pass and ((kZ >= 0) && (rtOut >= rtLo_point) && (rtOut <= rtHi_point));
    }

    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InOut = mdsInGPU.anchorRt[secondMDIndex];
    float sdIn_alpha = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);

    float tl_axis_x = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float tl_axis_y = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];

    betaIn = sdIn_alpha - SDL::phi_mpi_pi(acc, SDL::phi(acc, tl_axis_x, tl_axis_y) - mdsInGPU.anchorPhi[firstMDIndex]);

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
    betaInCut = alpaka::math::asin(
                    acc, alpaka::math::min(acc, (-sdIn_dr + dr) * SDL::k2Rinv1GeVf / SDL::ptCut, SDL::sinAlphaMax)) +
                (0.02f / sdIn_d);

    //Cut #4: first beta cut
    pass = pass and (alpaka::math::abs(acc, betaInRHmin) < betaInCut);
    return pass;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPointingConstraint(TAcc const& acc,
                                                             struct SDL::modules& modulesInGPU,
                                                             struct SDL::miniDoublets& mdsInGPU,
                                                             struct SDL::segments& segmentsInGPU,
                                                             uint16_t& innerInnerLowerModuleIndex,
                                                             uint16_t& middleLowerModuleIndex,
                                                             uint16_t& outerOuterLowerModuleIndex,
                                                             unsigned int& firstMDIndex,
                                                             unsigned int& secondMDIndex,
                                                             unsigned int& thirdMDIndex,
                                                             float& zOut,
                                                             float& rtOut,
                                                             uint16_t& innerOuterLowerModuleIndex,
                                                             unsigned int& innerSegmentIndex,
                                                             unsigned int& outerSegmentIndex,
                                                             float& betaIn,
                                                             float& betaInCut) {
    short innerInnerLowerModuleSubdet = modulesInGPU.subdets[innerInnerLowerModuleIndex];
    short middleLowerModuleSubdet = modulesInGPU.subdets[middleLowerModuleIndex];
    short outerOuterLowerModuleSubdet = modulesInGPU.subdets[outerOuterLowerModuleIndex];

    if (innerInnerLowerModuleSubdet == SDL::Barrel and middleLowerModuleSubdet == SDL::Barrel and
        outerOuterLowerModuleSubdet == SDL::Barrel) {
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
    } else if (innerInnerLowerModuleSubdet == SDL::Barrel and middleLowerModuleSubdet == SDL::Barrel and
               outerOuterLowerModuleSubdet == SDL::Endcap) {
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
    } else if (innerInnerLowerModuleSubdet == SDL::Barrel and middleLowerModuleSubdet == SDL::Endcap and
               outerOuterLowerModuleSubdet == SDL::Endcap) {
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

    else if (innerInnerLowerModuleSubdet == SDL::Endcap and middleLowerModuleSubdet == SDL::Endcap and
             outerOuterLowerModuleSubdet == SDL::Endcap) {
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
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computeRadiusFromThreeAnchorHits(
      TAcc const& acc, float x1, float y1, float x2, float y2, float x3, float y3, float& g, float& f) {
    float radius = 0.f;

    //writing manual code for computing radius, which obviously sucks
    //TODO:Use fancy inbuilt libraries like cuBLAS or cuSOLVE for this!
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
#ifdef Warnings
      printf("three collinear points or FATAL! r^2 < 0!\n");
#endif
      radius = -1.f;
    } else
      radius = alpaka::math::sqrt(acc, g * g + f * f - c);

    return radius;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runTripletConstraintsAndAlgo(TAcc const& acc,
                                                                   struct SDL::modules& modulesInGPU,
                                                                   struct SDL::miniDoublets& mdsInGPU,
                                                                   struct SDL::segments& segmentsInGPU,
                                                                   uint16_t& innerInnerLowerModuleIndex,
                                                                   uint16_t& middleLowerModuleIndex,
                                                                   uint16_t& outerOuterLowerModuleIndex,
                                                                   unsigned int& innerSegmentIndex,
                                                                   unsigned int& outerSegmentIndex,
                                                                   float& zOut,
                                                                   float& rtOut,
                                                                   float& deltaPhiPos,
                                                                   float& deltaPhi,
                                                                   float& betaIn,
                                                                   float& circleRadius,
                                                                   float& circleCenterX,
                                                                   float& circleCenterY,
                                                                   float& zLo,
                                                                   float& zHi,
                                                                   float& rtLo,
                                                                   float& rtHi,
                                                                   float& zLoPointed,
                                                                   float& zHiPointed,
                                                                   float& sdlCut,
                                                                   float& betaInCut) {
    bool pass = true;
    //this cut reduces the number of candidates by a factor of 4, i.e., 3 out of 4 warps can end right here!
    if (segmentsInGPU.mdIndices[2 * innerSegmentIndex + 1] != segmentsInGPU.mdIndices[2 * outerSegmentIndex])
      return false;

    unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex];
    unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex];
    unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex + 1];

    pass = pass and (passRZConstraint(acc,
                                      modulesInGPU,
                                      mdsInGPU,
                                      segmentsInGPU,
                                      innerInnerLowerModuleIndex,
                                      middleLowerModuleIndex,
                                      outerOuterLowerModuleIndex,
                                      firstMDIndex,
                                      secondMDIndex,
                                      thirdMDIndex));
    if (not pass)
      return pass;
    pass = pass and (passPointingConstraint(acc,
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
                                            betaInCut));
    if (not pass)
      return pass;

    float x1 = mdsInGPU.anchorX[firstMDIndex];
    float x2 = mdsInGPU.anchorX[secondMDIndex];
    float x3 = mdsInGPU.anchorX[thirdMDIndex];
    float y1 = mdsInGPU.anchorY[firstMDIndex];
    float y2 = mdsInGPU.anchorY[secondMDIndex];
    float y3 = mdsInGPU.anchorY[thirdMDIndex];

    circleRadius = computeRadiusFromThreeAnchorHits(acc, x1, y1, x2, y2, x3, y3, circleCenterX, circleCenterY);
    return pass;
  };

  struct createTripletsInGPUv2 {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  struct SDL::modules modulesInGPU,
                                  struct SDL::miniDoublets mdsInGPU,
                                  struct SDL::segments segmentsInGPU,
                                  struct SDL::triplets tripletsInGPU,
                                  struct SDL::objectRanges rangesInGPU,
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

            float zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, circleRadius, circleCenterX, circleCenterY;
            float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut;

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
                                                        deltaPhiPos,
                                                        deltaPhi,
                                                        betaIn,
                                                        circleRadius,
                                                        circleCenterX,
                                                        circleCenterY,
                                                        zLo,
                                                        zHi,
                                                        rtLo,
                                                        rtHi,
                                                        zLoPointed,
                                                        zHiPointed,
                                                        sdlCut,
                                                        betaInCut);

            if (success) {
              unsigned int totOccupancyTriplets = alpaka::atomicOp<alpaka::AtomicAdd>(
                  acc, &tripletsInGPU.totOccupancyTriplets[innerInnerLowerModuleIndex], 1u);
              if (static_cast<int>(totOccupancyTriplets) >=
                  rangesInGPU.tripletModuleOccupancy[innerInnerLowerModuleIndex]) {
#ifdef Warnings
                printf("Triplet excess alert! Module index = %d\n", innerInnerLowerModuleIndex);
#endif
              } else {
                unsigned int tripletModuleIndex =
                    alpaka::atomicOp<alpaka::AtomicAdd>(acc, &tripletsInGPU.nTriplets[innerInnerLowerModuleIndex], 1u);
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
                                   deltaPhiPos,
                                   deltaPhi,
                                   betaIn,
                                   circleRadius,
                                   circleCenterX,
                                   circleCenterY,
                                   zLo,
                                   zHi,
                                   rtLo,
                                   rtHi,
                                   zLoPointed,
                                   zHiPointed,
                                   sdlCut,
                                   betaInCut,
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

  struct createTripletArrayRanges {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  struct SDL::modules modulesInGPU,
                                  struct SDL::objectRanges rangesInGPU,
                                  struct SDL::segments segmentsInGPU) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      // Initialize variables in shared memory and set to 0
      int& nTotalTriplets = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      nTotalTriplets = 0;
      alpaka::syncBlockThreads(acc);

      // Initialize variables outside of the for loop.
      int occupancy, category_number, eta_number;

      for (uint16_t i = globalThreadIdx[2]; i < *modulesInGPU.nLowerModules; i += gridThreadExtent[2]) {
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
#ifdef Warnings
          printf("Unhandled case in createTripletArrayRanges! Module index = %i\n", i);
#endif
        }

        rangesInGPU.tripletModuleOccupancy[i] = occupancy;
        unsigned int nTotT = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &nTotalTriplets, occupancy);
        rangesInGPU.tripletModuleIndices[i] = nTotT;
      }

      // Wait for all threads to finish before reporting final values
      alpaka::syncBlockThreads(acc);
      if (globalThreadIdx[2] == 0) {
        *rangesInGPU.device_nTotalTrips = nTotalTriplets;
      }
    }
  };

  struct addTripletRangesToEventExplicit {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  struct SDL::modules modulesInGPU,
                                  struct SDL::triplets tripletsInGPU,
                                  struct SDL::objectRanges rangesInGPU) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (uint16_t i = globalThreadIdx[2]; i < *modulesInGPU.nLowerModules; i += gridThreadExtent[2]) {
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
}  // namespace SDL
#endif
