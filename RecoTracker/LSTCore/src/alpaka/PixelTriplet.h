#ifndef PixelTriplet_cuh
#define PixelTriplet_cuh

#ifdef LST_IS_CMSSW_PACKAGE
#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/alpaka/Module.h"
#else
#include "Constants.h"
#include "Module.h"
#endif

#include "Triplet.h"
#include "Segment.h"
#include "MiniDoublet.h"
#include "Hit.h"
#include "Quintuplet.h"

namespace SDL {
  // One pixel segment, one outer tracker triplet!
  struct pixelTriplets {
    unsigned int* pixelSegmentIndices;
    unsigned int* tripletIndices;
    unsigned int* nPixelTriplets;
    unsigned int* totOccupancyPixelTriplets;

    float* pixelRadiusError;
    float* rPhiChiSquared;
    float* rPhiChiSquaredInwards;
    float* rzChiSquared;

    FPX* pixelRadius;
    FPX* tripletRadius;
    FPX* pt;
    FPX* eta;
    FPX* phi;
    FPX* eta_pix;
    FPX* phi_pix;
    FPX* score;
    bool* isDup;
    bool* partOfPT5;

    uint8_t* logicalLayers;
    unsigned int* hitIndices;
    uint16_t* lowerModuleIndices;
    FPX* centerX;
    FPX* centerY;

    template <typename TBuff>
    void setData(TBuff& pixelTripletsBuffer) {
      pixelSegmentIndices = alpaka::getPtrNative(pixelTripletsBuffer.pixelSegmentIndices_buf);
      tripletIndices = alpaka::getPtrNative(pixelTripletsBuffer.tripletIndices_buf);
      nPixelTriplets = alpaka::getPtrNative(pixelTripletsBuffer.nPixelTriplets_buf);
      totOccupancyPixelTriplets = alpaka::getPtrNative(pixelTripletsBuffer.totOccupancyPixelTriplets_buf);
      pixelRadius = alpaka::getPtrNative(pixelTripletsBuffer.pixelRadius_buf);
      tripletRadius = alpaka::getPtrNative(pixelTripletsBuffer.tripletRadius_buf);
      pt = alpaka::getPtrNative(pixelTripletsBuffer.pt_buf);
      eta = alpaka::getPtrNative(pixelTripletsBuffer.eta_buf);
      phi = alpaka::getPtrNative(pixelTripletsBuffer.phi_buf);
      eta_pix = alpaka::getPtrNative(pixelTripletsBuffer.eta_pix_buf);
      phi_pix = alpaka::getPtrNative(pixelTripletsBuffer.phi_pix_buf);
      score = alpaka::getPtrNative(pixelTripletsBuffer.score_buf);
      isDup = alpaka::getPtrNative(pixelTripletsBuffer.isDup_buf);
      partOfPT5 = alpaka::getPtrNative(pixelTripletsBuffer.partOfPT5_buf);
      logicalLayers = alpaka::getPtrNative(pixelTripletsBuffer.logicalLayers_buf);
      hitIndices = alpaka::getPtrNative(pixelTripletsBuffer.hitIndices_buf);
      lowerModuleIndices = alpaka::getPtrNative(pixelTripletsBuffer.lowerModuleIndices_buf);
      centerX = alpaka::getPtrNative(pixelTripletsBuffer.centerX_buf);
      centerY = alpaka::getPtrNative(pixelTripletsBuffer.centerY_buf);
      pixelRadiusError = alpaka::getPtrNative(pixelTripletsBuffer.pixelRadiusError_buf);
      rPhiChiSquared = alpaka::getPtrNative(pixelTripletsBuffer.rPhiChiSquared_buf);
      rPhiChiSquaredInwards = alpaka::getPtrNative(pixelTripletsBuffer.rPhiChiSquaredInwards_buf);
      rzChiSquared = alpaka::getPtrNative(pixelTripletsBuffer.rzChiSquared_buf);
    }
  };

  template <typename TDev>
  struct pixelTripletsBuffer : pixelTriplets {
    Buf<TDev, unsigned int> pixelSegmentIndices_buf;
    Buf<TDev, unsigned int> tripletIndices_buf;
    Buf<TDev, unsigned int> nPixelTriplets_buf;
    Buf<TDev, unsigned int> totOccupancyPixelTriplets_buf;
    Buf<TDev, FPX> pixelRadius_buf;
    Buf<TDev, FPX> tripletRadius_buf;
    Buf<TDev, FPX> pt_buf;
    Buf<TDev, FPX> eta_buf;
    Buf<TDev, FPX> phi_buf;
    Buf<TDev, FPX> eta_pix_buf;
    Buf<TDev, FPX> phi_pix_buf;
    Buf<TDev, FPX> score_buf;
    Buf<TDev, bool> isDup_buf;
    Buf<TDev, bool> partOfPT5_buf;
    Buf<TDev, uint8_t> logicalLayers_buf;
    Buf<TDev, unsigned int> hitIndices_buf;
    Buf<TDev, uint16_t> lowerModuleIndices_buf;
    Buf<TDev, FPX> centerX_buf;
    Buf<TDev, FPX> centerY_buf;
    Buf<TDev, float> pixelRadiusError_buf;
    Buf<TDev, float> rPhiChiSquared_buf;
    Buf<TDev, float> rPhiChiSquaredInwards_buf;
    Buf<TDev, float> rzChiSquared_buf;

    template <typename TQueue, typename TDevAcc>
    pixelTripletsBuffer(unsigned int maxPixelTriplets, TDevAcc const& devAccIn, TQueue& queue)
        : pixelSegmentIndices_buf(allocBufWrapper<unsigned int>(devAccIn, maxPixelTriplets, queue)),
          tripletIndices_buf(allocBufWrapper<unsigned int>(devAccIn, maxPixelTriplets, queue)),
          nPixelTriplets_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          totOccupancyPixelTriplets_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          pixelRadius_buf(allocBufWrapper<FPX>(devAccIn, maxPixelTriplets, queue)),
          tripletRadius_buf(allocBufWrapper<FPX>(devAccIn, maxPixelTriplets, queue)),
          pt_buf(allocBufWrapper<FPX>(devAccIn, maxPixelTriplets, queue)),
          eta_buf(allocBufWrapper<FPX>(devAccIn, maxPixelTriplets, queue)),
          phi_buf(allocBufWrapper<FPX>(devAccIn, maxPixelTriplets, queue)),
          eta_pix_buf(allocBufWrapper<FPX>(devAccIn, maxPixelTriplets, queue)),
          phi_pix_buf(allocBufWrapper<FPX>(devAccIn, maxPixelTriplets, queue)),
          score_buf(allocBufWrapper<FPX>(devAccIn, maxPixelTriplets, queue)),
          isDup_buf(allocBufWrapper<bool>(devAccIn, maxPixelTriplets, queue)),
          partOfPT5_buf(allocBufWrapper<bool>(devAccIn, maxPixelTriplets, queue)),
          logicalLayers_buf(allocBufWrapper<uint8_t>(devAccIn, maxPixelTriplets * 5, queue)),
          hitIndices_buf(allocBufWrapper<unsigned int>(devAccIn, maxPixelTriplets * 10, queue)),
          lowerModuleIndices_buf(allocBufWrapper<uint16_t>(devAccIn, maxPixelTriplets * 5, queue)),
          centerX_buf(allocBufWrapper<FPX>(devAccIn, maxPixelTriplets, queue)),
          centerY_buf(allocBufWrapper<FPX>(devAccIn, maxPixelTriplets, queue)),
          pixelRadiusError_buf(allocBufWrapper<float>(devAccIn, maxPixelTriplets, queue)),
          rPhiChiSquared_buf(allocBufWrapper<float>(devAccIn, maxPixelTriplets, queue)),
          rPhiChiSquaredInwards_buf(allocBufWrapper<float>(devAccIn, maxPixelTriplets, queue)),
          rzChiSquared_buf(allocBufWrapper<float>(devAccIn, maxPixelTriplets, queue)) {
      alpaka::memset(queue, nPixelTriplets_buf, 0u);
      alpaka::memset(queue, totOccupancyPixelTriplets_buf, 0u);
      alpaka::memset(queue, partOfPT5_buf, false);
      alpaka::wait(queue);
    }
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addPixelTripletToMemory(struct SDL::modules& modulesInGPU,
                                                              struct SDL::miniDoublets& mdsInGPU,
                                                              struct SDL::segments& segmentsInGPU,
                                                              struct SDL::triplets& tripletsInGPU,
                                                              struct SDL::pixelTriplets& pixelTripletsInGPU,
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
    pixelTripletsInGPU.pixelSegmentIndices[pixelTripletIndex] = pixelSegmentIndex;
    pixelTripletsInGPU.tripletIndices[pixelTripletIndex] = tripletIndex;
    pixelTripletsInGPU.pixelRadius[pixelTripletIndex] = __F2H(pixelRadius);
    pixelTripletsInGPU.tripletRadius[pixelTripletIndex] = __F2H(tripletRadius);
    pixelTripletsInGPU.pt[pixelTripletIndex] = __F2H(pt);
    pixelTripletsInGPU.eta[pixelTripletIndex] = __F2H(eta);
    pixelTripletsInGPU.phi[pixelTripletIndex] = __F2H(phi);
    pixelTripletsInGPU.eta_pix[pixelTripletIndex] = __F2H(eta_pix);
    pixelTripletsInGPU.phi_pix[pixelTripletIndex] = __F2H(phi_pix);
    pixelTripletsInGPU.isDup[pixelTripletIndex] = false;
    pixelTripletsInGPU.score[pixelTripletIndex] = __F2H(score);

    pixelTripletsInGPU.centerX[pixelTripletIndex] = __F2H(centerX);
    pixelTripletsInGPU.centerY[pixelTripletIndex] = __F2H(centerY);
    pixelTripletsInGPU.logicalLayers[5 * pixelTripletIndex] = 0;
    pixelTripletsInGPU.logicalLayers[5 * pixelTripletIndex + 1] = 0;
    pixelTripletsInGPU.logicalLayers[5 * pixelTripletIndex + 2] = tripletsInGPU.logicalLayers[tripletIndex * 3];
    pixelTripletsInGPU.logicalLayers[5 * pixelTripletIndex + 3] = tripletsInGPU.logicalLayers[tripletIndex * 3 + 1];
    pixelTripletsInGPU.logicalLayers[5 * pixelTripletIndex + 4] = tripletsInGPU.logicalLayers[tripletIndex * 3 + 2];

    pixelTripletsInGPU.lowerModuleIndices[5 * pixelTripletIndex] =
        segmentsInGPU.innerLowerModuleIndices[pixelSegmentIndex];
    pixelTripletsInGPU.lowerModuleIndices[5 * pixelTripletIndex + 1] =
        segmentsInGPU.outerLowerModuleIndices[pixelSegmentIndex];
    pixelTripletsInGPU.lowerModuleIndices[5 * pixelTripletIndex + 2] =
        tripletsInGPU.lowerModuleIndices[3 * tripletIndex];
    pixelTripletsInGPU.lowerModuleIndices[5 * pixelTripletIndex + 3] =
        tripletsInGPU.lowerModuleIndices[3 * tripletIndex + 1];
    pixelTripletsInGPU.lowerModuleIndices[5 * pixelTripletIndex + 4] =
        tripletsInGPU.lowerModuleIndices[3 * tripletIndex + 2];

    unsigned int pixelInnerMD = segmentsInGPU.mdIndices[2 * pixelSegmentIndex];
    unsigned int pixelOuterMD = segmentsInGPU.mdIndices[2 * pixelSegmentIndex + 1];

    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex] = mdsInGPU.anchorHitIndices[pixelInnerMD];
    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 1] = mdsInGPU.outerHitIndices[pixelInnerMD];
    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 2] = mdsInGPU.anchorHitIndices[pixelOuterMD];
    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 3] = mdsInGPU.outerHitIndices[pixelOuterMD];

    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 4] = tripletsInGPU.hitIndices[6 * tripletIndex];
    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 5] = tripletsInGPU.hitIndices[6 * tripletIndex + 1];
    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 6] = tripletsInGPU.hitIndices[6 * tripletIndex + 2];
    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 7] = tripletsInGPU.hitIndices[6 * tripletIndex + 3];
    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 8] = tripletsInGPU.hitIndices[6 * tripletIndex + 4];
    pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex + 9] = tripletsInGPU.hitIndices[6 * tripletIndex + 5];
    pixelTripletsInGPU.rPhiChiSquared[pixelTripletIndex] = rPhiChiSquared;
    pixelTripletsInGPU.rPhiChiSquaredInwards[pixelTripletIndex] = rPhiChiSquaredInwards;
    pixelTripletsInGPU.rzChiSquared[pixelTripletIndex] = rzChiSquared;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runPixelTrackletDefaultAlgopT3(TAcc const& acc,
                                                                     struct SDL::modules& modulesInGPU,
                                                                     struct SDL::objectRanges& rangesInGPU,
                                                                     struct SDL::miniDoublets& mdsInGPU,
                                                                     struct SDL::segments& segmentsInGPU,
                                                                     uint16_t& pixelLowerModuleIndex,
                                                                     uint16_t& outerInnerLowerModuleIndex,
                                                                     uint16_t& outerOuterLowerModuleIndex,
                                                                     unsigned int& innerSegmentIndex,
                                                                     unsigned int& outerSegmentIndex,
                                                                     float& zOut,
                                                                     float& rtOut,
                                                                     float& deltaPhiPos,
                                                                     float& deltaPhi,
                                                                     float& betaIn,
                                                                     float& betaOut,
                                                                     float& pt_beta,
                                                                     float& zLo,
                                                                     float& zHi,
                                                                     float& rtLo,
                                                                     float& rtHi,
                                                                     float& zLoPointed,
                                                                     float& zHiPointed,
                                                                     float& sdlCut,
                                                                     float& betaInCut,
                                                                     float& betaOutCut,
                                                                     float& deltaBetaCut,
                                                                     float& kZ) {
    zLo = -999;
    zHi = -999;
    rtLo = -999;
    rtHi = -999;
    zLoPointed = -999;
    zHiPointed = -999;
    kZ = -999;
    betaInCut = -999;

    short outerInnerLowerModuleSubdet = modulesInGPU.subdets[outerInnerLowerModuleIndex];
    short outerOuterLowerModuleSubdet = modulesInGPU.subdets[outerOuterLowerModuleIndex];

    unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex];
    unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex + 1];

    unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex];
    unsigned int fourthMDIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex + 1];

    if (outerInnerLowerModuleSubdet == SDL::Barrel and
        (outerOuterLowerModuleSubdet == SDL::Barrel or outerOuterLowerModuleSubdet == SDL::Endcap)) {
      return runTripletDefaultAlgoPPBB(acc,
                                       modulesInGPU,
                                       rangesInGPU,
                                       mdsInGPU,
                                       segmentsInGPU,
                                       pixelLowerModuleIndex,
                                       outerInnerLowerModuleIndex,
                                       outerOuterLowerModuleIndex,
                                       innerSegmentIndex,
                                       outerSegmentIndex,
                                       firstMDIndex,
                                       secondMDIndex,
                                       thirdMDIndex,
                                       fourthMDIndex,
                                       zOut,
                                       rtOut,
                                       deltaPhiPos,
                                       deltaPhi,
                                       betaIn,
                                       betaOut,
                                       pt_beta,
                                       zLo,
                                       zHi,
                                       zLoPointed,
                                       zHiPointed,
                                       sdlCut,
                                       betaOutCut,
                                       deltaBetaCut);
    } else if (outerInnerLowerModuleSubdet == SDL::Endcap and outerOuterLowerModuleSubdet == SDL::Endcap) {
      return runTripletDefaultAlgoPPEE(acc,
                                       modulesInGPU,
                                       rangesInGPU,
                                       mdsInGPU,
                                       segmentsInGPU,
                                       pixelLowerModuleIndex,
                                       outerInnerLowerModuleIndex,
                                       outerOuterLowerModuleIndex,
                                       innerSegmentIndex,
                                       outerSegmentIndex,
                                       firstMDIndex,
                                       secondMDIndex,
                                       thirdMDIndex,
                                       fourthMDIndex,
                                       zOut,
                                       rtOut,
                                       deltaPhiPos,
                                       deltaPhi,
                                       betaIn,
                                       betaOut,
                                       pt_beta,
                                       zLo,
                                       rtLo,
                                       rtHi,
                                       sdlCut,
                                       betaInCut,
                                       betaOutCut,
                                       deltaBetaCut,
                                       kZ);
    }
    return false;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT3RZChiSquaredCuts(struct SDL::modules& modulesInGPU,
                                                              uint16_t& lowerModuleIndex1,
                                                              uint16_t& lowerModuleIndex2,
                                                              uint16_t& lowerModuleIndex3,
                                                              float& rzChiSquared) {
    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);

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
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computeChiSquaredpT3(TAcc const& acc,
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
    //given values of (g, f, radius) and a set of points (and its uncertainties)
    //compute chi squared
    float c = g * g + f * f - radius * radius;
    float chiSquared = 0.f;
    float absArctanSlope, angleM, xPrime, yPrime, sigma;
    for (size_t i = 0; i < nPoints; i++) {
      absArctanSlope = ((slopes[i] != SDL::SDL_INF) ? alpaka::math::abs(acc, alpaka::math::atan(acc, slopes[i]))
                                                    : 0.5f * float(M_PI));
      if (xs[i] > 0 and ys[i] > 0) {
        angleM = 0.5f * float(M_PI) - absArctanSlope;
      } else if (xs[i] < 0 and ys[i] > 0) {
        angleM = absArctanSlope + 0.5f * float(M_PI);
      } else if (xs[i] < 0 and ys[i] < 0) {
        angleM = -(absArctanSlope + 0.5f * float(M_PI));
      } else if (xs[i] > 0 and ys[i] < 0) {
        angleM = -(0.5f * float(M_PI) - absArctanSlope);
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
      sigma = 2 * alpaka::math::sqrt(
                      acc, (xPrime * delta1[i]) * (xPrime * delta1[i]) + (yPrime * delta2[i]) * (yPrime * delta2[i]));
      chiSquared += (xs[i] * xs[i] + ys[i] * ys[i] - 2 * g * xs[i] - 2 * f * ys[i] + c) *
                    (xs[i] * xs[i] + ys[i] * ys[i] - 2 * g * xs[i] - 2 * f * ys[i] + c) / (sigma * sigma);
    }
    return chiSquared;
  };

  //TODO: merge this one and the pT5 function later into a single function
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computePT3RPhiChiSquared(TAcc const& acc,
                                                                struct SDL::modules& modulesInGPU,
                                                                uint16_t* lowerModuleIndices,
                                                                float& g,
                                                                float& f,
                                                                float& radius,
                                                                float* xs,
                                                                float* ys) {
    float delta1[3]{}, delta2[3]{}, slopes[3];
    bool isFlat[3]{};
    float chiSquared = 0;
    float inv1 = 0.01f / 0.009f;
    float inv2 = 0.15f / 0.009f;
    for (size_t i = 0; i < 3; i++) {
      ModuleType moduleType = modulesInGPU.moduleType[lowerModuleIndices[i]];
      short moduleSubdet = modulesInGPU.subdets[lowerModuleIndices[i]];
      short moduleSide = modulesInGPU.sides[lowerModuleIndices[i]];
      float drdz = modulesInGPU.drdzs[lowerModuleIndices[i]];
      slopes[i] = modulesInGPU.dxdys[lowerModuleIndices[i]];
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
#ifdef Warnings
      else {
        printf("ERROR!!!!! I SHOULDN'T BE HERE!!!! subdet = %d, type = %d, side = %d\n",
               moduleSubdet,
               moduleType,
               moduleSide);
      }
#endif
    }
    chiSquared = computeChiSquaredpT3(acc, 3, xs, ys, delta1, delta2, slopes, isFlat, g, f, radius);

    return chiSquared;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computePT3RPhiChiSquaredInwards(
      struct SDL::modules& modulesInGPU, float& g, float& f, float& r, float* xPix, float* yPix) {
    float residual = (xPix[0] - g) * (xPix[0] - g) + (yPix[0] - f) * (yPix[0] - f) - r * r;
    float chiSquared = residual * residual;
    residual = (xPix[1] - g) * (xPix[1] - g) + (yPix[1] - f) * (yPix[1] - f) - r * r;
    chiSquared += residual * residual;

    chiSquared *= 0.5f;
    return chiSquared;
  };

  //90pc threshold
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT3RPhiChiSquaredCuts(struct SDL::modules& modulesInGPU,
                                                                uint16_t& lowerModuleIndex1,
                                                                uint16_t& lowerModuleIndex2,
                                                                uint16_t& lowerModuleIndex3,
                                                                float& chiSquared) {
    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);

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
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT3RPhiChiSquaredInwardsCuts(struct SDL::modules& modulesInGPU,
                                                                       uint16_t& lowerModuleIndex1,
                                                                       uint16_t& lowerModuleIndex2,
                                                                       uint16_t& lowerModuleIndex3,
                                                                       float& chiSquared) {
    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);

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
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool checkIntervalOverlappT3(const float& firstMin,
                                                              const float& firstMax,
                                                              const float& secondMin,
                                                              const float& secondMax) {
    return ((firstMin <= secondMin) && (secondMin < firstMax)) || ((secondMin < firstMin) && (firstMin < secondMax));
  };

  /*bounds for high Pt taken from : http://uaf-10.t2.ucsd.edu/~bsathian/SDL/T5_efficiency/efficiencies/new_efficiencies/efficiencies_20210513_T5_recovering_high_Pt_efficiencies/highE_radius_matching/highE_bounds.txt */
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passRadiusCriterionBBB(TAcc const& acc,
                                                             float const& pixelRadius,
                                                             float const& pixelRadiusError,
                                                             float const& tripletRadius) {
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
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passRadiusCriterionBBE(TAcc const& acc,
                                                             float const& pixelRadius,
                                                             float const& pixelRadiusError,
                                                             float const& tripletRadius) {
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
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passRadiusCriterionBEE(TAcc const& acc,
                                                             float const& pixelRadius,
                                                             float const& pixelRadiusError,
                                                             float const& tripletRadius) {
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
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passRadiusCriterionEEE(TAcc const& acc,
                                                             float const& pixelRadius,
                                                             float const& pixelRadiusError,
                                                             float const& tripletRadius) {
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
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passRadiusCriterion(TAcc const& acc,
                                                          struct SDL::modules const& modulesInGPU,
                                                          float const& pixelRadius,
                                                          float const& pixelRadiusError,
                                                          float const& tripletRadius,
                                                          uint16_t const& lowerModuleIndex,
                                                          uint16_t const& middleModuleIndex,
                                                          uint16_t const& upperModuleIndex) {
    if (modulesInGPU.subdets[lowerModuleIndex] == SDL::Endcap) {
      return passRadiusCriterionEEE(acc, pixelRadius, pixelRadiusError, tripletRadius);
    } else if (modulesInGPU.subdets[middleModuleIndex] == SDL::Endcap) {
      return passRadiusCriterionBEE(acc, pixelRadius, pixelRadiusError, tripletRadius);
    } else if (modulesInGPU.subdets[upperModuleIndex] == SDL::Endcap) {
      return passRadiusCriterionBBE(acc, pixelRadius, pixelRadiusError, tripletRadius);
    } else {
      return passRadiusCriterionBBB(acc, pixelRadius, pixelRadiusError, tripletRadius);
    }
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computePT3RZChiSquared(TAcc const& acc,
                                                              struct SDL::modules const& modulesInGPU,
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
    float error = 0;
    float RMSE = 0;

    float Px = pixelSegmentPx, Py = pixelSegmentPy, Pz = pixelSegmentPz;
    int charge = pixelSegmentCharge;
    float x1 = xPix[1] / 100;
    float y1 = yPix[1] / 100;
    float z1 = zPix[1] / 100;
    float r1 = rtPix[1] / 100;

    float Bz = SDL::magnetic_field;
    float a = -0.299792 * Bz * charge;

    for (size_t i = 0; i < 3; i++) {
      float zsi = zs[i] / 100;
      float rtsi = rts[i] / 100;
      uint16_t lowerModuleIndex = lowerModuleIndices[i];
      const int moduleType = modulesInGPU.moduleType[lowerModuleIndex];
      const int moduleSide = modulesInGPU.sides[lowerModuleIndex];
      const int moduleSubdet = modulesInGPU.subdets[lowerModuleIndex];

      // calculation is detailed documented here https://indico.cern.ch/event/1185895/contributions/4982756/attachments/2526561/4345805/helix%20pT3%20summarize.pdf
      float diffr, diffz;
      float p = alpaka::math::sqrt(acc, Px * Px + Py * Py + Pz * Pz);

      float rou = a / p;
      if (moduleSubdet == SDL::Endcap) {
        float s = (zsi - z1) * p / Pz;
        float x = x1 + Px / a * alpaka::math::sin(acc, rou * s) - Py / a * (1 - alpaka::math::cos(acc, rou * s));
        float y = y1 + Py / a * alpaka::math::sin(acc, rou * s) + Px / a * (1 - alpaka::math::cos(acc, rou * s));
        diffr = alpaka::math::abs(acc, rtsi - alpaka::math::sqrt(acc, x * x + y * y)) * 100;
      }

      if (moduleSubdet == SDL::Barrel) {
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

      residual = moduleSubdet == SDL::Barrel ? diffz : diffr;

      //PS Modules
      if (moduleType == 0) {
        error = 0.15f;
      } else  //2S modules
      {
        error = 5.0f;
      }

      //special dispensation to tilted PS modules!
      if (moduleType == 0 and moduleSubdet == SDL::Barrel and moduleSide != Center) {
        float drdz = modulesInGPU.drdzs[lowerModuleIndex];
        error /= alpaka::math::sqrt(acc, 1 + drdz * drdz);
      }
      RMSE += (residual * residual) / (error * error);
    }

    RMSE = alpaka::math::sqrt(acc, 0.2f * RMSE);  //the constant doesn't really matter....

    return RMSE;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runPixelTripletDefaultAlgo(TAcc const& acc,
                                                                 struct SDL::modules& modulesInGPU,
                                                                 struct SDL::objectRanges& rangesInGPU,
                                                                 struct SDL::miniDoublets& mdsInGPU,
                                                                 struct SDL::segments& segmentsInGPU,
                                                                 struct SDL::triplets& tripletsInGPU,
                                                                 unsigned int& pixelSegmentIndex,
                                                                 unsigned int tripletIndex,
                                                                 float& pixelRadius,
                                                                 float& pixelRadiusError,
                                                                 float& tripletRadius,
                                                                 float& centerX,
                                                                 float& centerY,
                                                                 float& rzChiSquared,
                                                                 float& rPhiChiSquared,
                                                                 float& rPhiChiSquaredInwards,
                                                                 bool runChiSquaredCuts = true) {
    bool pass = true;

    //run pT4 compatibility between the pixel segment and inner segment, and between the pixel and outer segment of the triplet
    uint16_t pixelModuleIndex = segmentsInGPU.innerLowerModuleIndices[pixelSegmentIndex];

    uint16_t lowerModuleIndex = tripletsInGPU.lowerModuleIndices[3 * tripletIndex];
    uint16_t middleModuleIndex = tripletsInGPU.lowerModuleIndices[3 * tripletIndex + 1];
    uint16_t upperModuleIndex = tripletsInGPU.lowerModuleIndices[3 * tripletIndex + 2];

    {
      //placeholder
      float zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta;  //temp stuff
      float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;

      // pixel segment vs inner segment of the triplet
      pass = pass and runPixelTrackletDefaultAlgopT3(acc,
                                                     modulesInGPU,
                                                     rangesInGPU,
                                                     mdsInGPU,
                                                     segmentsInGPU,
                                                     pixelModuleIndex,
                                                     lowerModuleIndex,
                                                     middleModuleIndex,
                                                     pixelSegmentIndex,
                                                     tripletsInGPU.segmentIndices[2 * tripletIndex],
                                                     zOut,
                                                     rtOut,
                                                     deltaPhiPos,
                                                     deltaPhi,
                                                     betaIn,
                                                     betaOut,
                                                     pt_beta,
                                                     zLo,
                                                     zHi,
                                                     rtLo,
                                                     rtHi,
                                                     zLoPointed,
                                                     zHiPointed,
                                                     sdlCut,
                                                     betaInCut,
                                                     betaOutCut,
                                                     deltaBetaCut,
                                                     kZ);
      if (not pass)
        return pass;

      //pixel segment vs outer segment of triplet
      pass = pass and runPixelTrackletDefaultAlgopT3(acc,
                                                     modulesInGPU,
                                                     rangesInGPU,
                                                     mdsInGPU,
                                                     segmentsInGPU,
                                                     pixelModuleIndex,
                                                     middleModuleIndex,
                                                     upperModuleIndex,
                                                     pixelSegmentIndex,
                                                     tripletsInGPU.segmentIndices[2 * tripletIndex + 1],
                                                     zOut,
                                                     rtOut,
                                                     deltaPhiPos,
                                                     deltaPhi,
                                                     betaIn,
                                                     betaOut,
                                                     pt_beta,
                                                     zLo,
                                                     zHi,
                                                     rtLo,
                                                     rtHi,
                                                     zLoPointed,
                                                     zHiPointed,
                                                     sdlCut,
                                                     betaInCut,
                                                     betaOutCut,
                                                     deltaBetaCut,
                                                     kZ);
      if (not pass)
        return pass;
    }

    //pt matching between the pixel ptin and the triplet circle pt
    unsigned int pixelSegmentArrayIndex = pixelSegmentIndex - rangesInGPU.segmentModuleIndices[pixelModuleIndex];
    float pixelSegmentPt = segmentsInGPU.ptIn[pixelSegmentArrayIndex];
    float pixelSegmentPtError = segmentsInGPU.ptErr[pixelSegmentArrayIndex];
    float pixelSegmentPx = segmentsInGPU.px[pixelSegmentArrayIndex];
    float pixelSegmentPy = segmentsInGPU.py[pixelSegmentArrayIndex];
    float pixelSegmentPz = segmentsInGPU.pz[pixelSegmentArrayIndex];
    int pixelSegmentCharge = segmentsInGPU.charge[pixelSegmentArrayIndex];

    float pixelG = segmentsInGPU.circleCenterX[pixelSegmentArrayIndex];
    float pixelF = segmentsInGPU.circleCenterY[pixelSegmentArrayIndex];
    float pixelRadiusPCA = segmentsInGPU.circleRadius[pixelSegmentArrayIndex];

    unsigned int pixelInnerMDIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex];
    unsigned int pixelOuterMDIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex + 1];

    pixelRadius = pixelSegmentPt * kR1GeVf;
    pixelRadiusError = pixelSegmentPtError * kR1GeVf;
    unsigned int tripletInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * tripletIndex];
    unsigned int tripletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * tripletIndex + 1];

    unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 * tripletInnerSegmentIndex];
    unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * tripletInnerSegmentIndex + 1];
    unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * tripletOuterSegmentIndex + 1];

    float xs[3] = {mdsInGPU.anchorX[firstMDIndex], mdsInGPU.anchorX[secondMDIndex], mdsInGPU.anchorX[thirdMDIndex]};
    float ys[3] = {mdsInGPU.anchorY[firstMDIndex], mdsInGPU.anchorY[secondMDIndex], mdsInGPU.anchorY[thirdMDIndex]};

    float g, f;
    tripletRadius = tripletsInGPU.circleRadius[tripletIndex];
    g = tripletsInGPU.circleCenterX[tripletIndex];
    f = tripletsInGPU.circleCenterY[tripletIndex];

    pass = pass and passRadiusCriterion(acc,
                                        modulesInGPU,
                                        pixelRadius,
                                        pixelRadiusError,
                                        tripletRadius,
                                        lowerModuleIndex,
                                        middleModuleIndex,
                                        upperModuleIndex);
    if (not pass)
      return pass;

    uint16_t lowerModuleIndices[3] = {lowerModuleIndex, middleModuleIndex, upperModuleIndex};

    if (runChiSquaredCuts and pixelSegmentPt < 5.0f) {
      float rts[3] = {
          mdsInGPU.anchorRt[firstMDIndex], mdsInGPU.anchorRt[secondMDIndex], mdsInGPU.anchorRt[thirdMDIndex]};
      float zs[3] = {mdsInGPU.anchorZ[firstMDIndex], mdsInGPU.anchorZ[secondMDIndex], mdsInGPU.anchorZ[thirdMDIndex]};
      float rtPix[2] = {mdsInGPU.anchorRt[pixelInnerMDIndex], mdsInGPU.anchorRt[pixelOuterMDIndex]};
      float xPix[2] = {mdsInGPU.anchorX[pixelInnerMDIndex], mdsInGPU.anchorX[pixelOuterMDIndex]};
      float yPix[2] = {mdsInGPU.anchorY[pixelInnerMDIndex], mdsInGPU.anchorY[pixelOuterMDIndex]};
      float zPix[2] = {mdsInGPU.anchorZ[pixelInnerMDIndex], mdsInGPU.anchorZ[pixelOuterMDIndex]};

      rzChiSquared = computePT3RZChiSquared(acc,
                                            modulesInGPU,
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
      pass = pass and
             passPT3RZChiSquaredCuts(modulesInGPU, lowerModuleIndex, middleModuleIndex, upperModuleIndex, rzChiSquared);
      if (not pass)
        return pass;
    } else {
      rzChiSquared = -1;
    }

    rPhiChiSquared =
        computePT3RPhiChiSquared(acc, modulesInGPU, lowerModuleIndices, pixelG, pixelF, pixelRadiusPCA, xs, ys);

    if (runChiSquaredCuts and pixelSegmentPt < 5.0f) {
      pass = pass and passPT3RPhiChiSquaredCuts(
                          modulesInGPU, lowerModuleIndex, middleModuleIndex, upperModuleIndex, rPhiChiSquared);
      if (not pass)
        return pass;
    }

    float xPix[2] = {mdsInGPU.anchorX[pixelInnerMDIndex], mdsInGPU.anchorX[pixelOuterMDIndex]};
    float yPix[2] = {mdsInGPU.anchorY[pixelInnerMDIndex], mdsInGPU.anchorY[pixelOuterMDIndex]};
    rPhiChiSquaredInwards = computePT3RPhiChiSquaredInwards(modulesInGPU, g, f, tripletRadius, xPix, yPix);

    if (runChiSquaredCuts and pixelSegmentPt < 5.0f) {
      pass = pass and passPT3RPhiChiSquaredInwardsCuts(
                          modulesInGPU, lowerModuleIndex, middleModuleIndex, upperModuleIndex, rPhiChiSquaredInwards);
      if (not pass)
        return pass;
    }
    centerX = 0;
    centerY = 0;
    return pass;
  };

  struct createPixelTripletsInGPUFromMapv2 {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  struct SDL::modules modulesInGPU,
                                  struct SDL::objectRanges rangesInGPU,
                                  struct SDL::miniDoublets mdsInGPU,
                                  struct SDL::segments segmentsInGPU,
                                  struct SDL::triplets tripletsInGPU,
                                  struct SDL::pixelTriplets pixelTripletsInGPU,
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
              modulesInGPU
                  .connectedPixels[iLSModule];  //connected pixels will have the appropriate lower module index by default!
#ifdef Warnings
          if (tripletLowerModuleIndex >= *modulesInGPU.nLowerModules) {
            printf("tripletLowerModuleIndex %d >= modulesInGPU.nLowerModules %d \n",
                   tripletLowerModuleIndex,
                   *modulesInGPU.nLowerModules);
            continue;  //sanity check
          }
#endif
          //Removes 2S-2S :FIXME: filter these out in the pixel map
          if (modulesInGPU.moduleType[tripletLowerModuleIndex] == SDL::TwoS)
            continue;

          uint16_t pixelModuleIndex = *modulesInGPU.nLowerModules;
          unsigned int nOuterTriplets = tripletsInGPU.nTriplets[tripletLowerModuleIndex];
          if (nOuterTriplets == 0)
            continue;

          unsigned int pixelSegmentIndex = rangesInGPU.segmentModuleIndices[pixelModuleIndex] + i_pLS;

          if (segmentsInGPU.isDup[i_pLS])
            continue;
          if (segmentsInGPU.partOfPT5[i_pLS])
            continue;  //don't make pT3s for those pixels that are part of pT5

          short layer2_adjustment;
          if (modulesInGPU.layers[tripletLowerModuleIndex] == 1) {
            layer2_adjustment = 1;
          }  //get upper segment to be in second layer
          else if (modulesInGPU.layers[tripletLowerModuleIndex] == 2) {
            layer2_adjustment = 0;
          }  // get lower segment to be in second layer
          else {
            continue;
          }

          //fetch the triplet
          for (unsigned int outerTripletArrayIndex = globalThreadIdx[2]; outerTripletArrayIndex < nOuterTriplets;
               outerTripletArrayIndex += gridThreadExtent[2]) {
            unsigned int outerTripletIndex =
                rangesInGPU.tripletModuleIndices[tripletLowerModuleIndex] + outerTripletArrayIndex;
            if (modulesInGPU.moduleType[tripletsInGPU.lowerModuleIndices[3 * outerTripletIndex + 1]] == SDL::TwoS)
              continue;  //REMOVES PS-2S

            if (tripletsInGPU.partOfPT5[outerTripletIndex])
              continue;  //don't create pT3s for T3s accounted in pT5s

            float pixelRadius, pixelRadiusError, tripletRadius, rPhiChiSquared, rzChiSquared, rPhiChiSquaredInwards,
                centerX, centerY;
            bool success = runPixelTripletDefaultAlgo(acc,
                                                      modulesInGPU,
                                                      rangesInGPU,
                                                      mdsInGPU,
                                                      segmentsInGPU,
                                                      tripletsInGPU,
                                                      pixelSegmentIndex,
                                                      outerTripletIndex,
                                                      pixelRadius,
                                                      pixelRadiusError,
                                                      tripletRadius,
                                                      centerX,
                                                      centerY,
                                                      rzChiSquared,
                                                      rPhiChiSquared,
                                                      rPhiChiSquaredInwards);

            if (success) {
              float phi =
                  mdsInGPU.anchorPhi[segmentsInGPU.mdIndices[2 * tripletsInGPU.segmentIndices[2 * outerTripletIndex] +
                                                             layer2_adjustment]];
              float eta =
                  mdsInGPU.anchorEta[segmentsInGPU.mdIndices[2 * tripletsInGPU.segmentIndices[2 * outerTripletIndex] +
                                                             layer2_adjustment]];
              float eta_pix = segmentsInGPU.eta[i_pLS];
              float phi_pix = segmentsInGPU.phi[i_pLS];
              float pt = segmentsInGPU.ptIn[i_pLS];
              float score = rPhiChiSquared + rPhiChiSquaredInwards;
              unsigned int totOccupancyPixelTriplets =
                  alpaka::atomicOp<alpaka::AtomicAdd>(acc, pixelTripletsInGPU.totOccupancyPixelTriplets, 1u);
              if (totOccupancyPixelTriplets >= N_MAX_PIXEL_TRIPLETS) {
#ifdef Warnings
                printf("Pixel Triplet excess alert!\n");
#endif
              } else {
                unsigned int pixelTripletIndex =
                    alpaka::atomicOp<alpaka::AtomicAdd>(acc, pixelTripletsInGPU.nPixelTriplets, 1u);
                addPixelTripletToMemory(modulesInGPU,
                                        mdsInGPU,
                                        segmentsInGPU,
                                        tripletsInGPU,
                                        pixelTripletsInGPU,
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
                tripletsInGPU.partOfPT3[outerTripletIndex] = true;
              }
            }
          }  // for outerTripletArrayIndex
        }    // for iLSModule < iLSModule_max
      }      // for i_pLS
    }
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void runDeltaBetaIterationspT3(TAcc const& acc,
                                                                float& betaIn,
                                                                float& betaOut,
                                                                float& betaAv,
                                                                float& pt_beta,
                                                                float sdIn_dr,
                                                                float sdOut_dr,
                                                                float dr,
                                                                float lIn) {
    if (lIn == 0) {
      betaOut += SDL::copysignf(
          alpaka::math::asin(
              acc,
              alpaka::math::min(acc, sdOut_dr * SDL::k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), SDL::sinAlphaMax)),
          betaOut);
      return;
    }

    if (betaIn * betaOut > 0.f and
        (alpaka::math::abs(acc, pt_beta) < 4.f * SDL::pt_betaMax or
         (lIn >= 11 and alpaka::math::abs(acc, pt_beta) <
                            8.f * SDL::pt_betaMax)))  //and the pt_beta is well-defined; less strict for endcap-endcap
    {
      const float betaInUpd =
          betaIn +
          SDL::copysignf(alpaka::math::asin(
                             acc,
                             alpaka::math::min(
                                 acc, sdIn_dr * SDL::k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), SDL::sinAlphaMax)),
                         betaIn);  //FIXME: need a faster version
      const float betaOutUpd =
          betaOut +
          SDL::copysignf(alpaka::math::asin(
                             acc,
                             alpaka::math::min(
                                 acc, sdOut_dr * SDL::k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), SDL::sinAlphaMax)),
                         betaOut);  //FIXME: need a faster version
      betaAv = 0.5f * (betaInUpd + betaOutUpd);

      //1st update
      const float pt_beta_inv =
          1.f / alpaka::math::abs(acc, dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv));  //get a better pt estimate

      betaIn += SDL::copysignf(
          alpaka::math::asin(acc, alpaka::math::min(acc, sdIn_dr * SDL::k2Rinv1GeVf * pt_beta_inv, SDL::sinAlphaMax)),
          betaIn);  //FIXME: need a faster version
      betaOut += SDL::copysignf(
          alpaka::math::asin(acc, alpaka::math::min(acc, sdOut_dr * SDL::k2Rinv1GeVf * pt_beta_inv, SDL::sinAlphaMax)),
          betaOut);  //FIXME: need a faster version
      //update the av and pt
      betaAv = 0.5f * (betaIn + betaOut);
      //2nd update
      pt_beta = dr * SDL::k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);  //get a better pt estimate
    } else if (lIn < 11 && alpaka::math::abs(acc, betaOut) < 0.2f * alpaka::math::abs(acc, betaIn) &&
               alpaka::math::abs(acc, pt_beta) < 12.f * SDL::pt_betaMax)  //use betaIn sign as ref
    {
      const float pt_betaIn = dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaIn);

      const float betaInUpd =
          betaIn + SDL::copysignf(
                       alpaka::math::asin(
                           acc,
                           alpaka::math::min(
                               acc, sdIn_dr * SDL::k2Rinv1GeVf / alpaka::math::abs(acc, pt_betaIn), SDL::sinAlphaMax)),
                       betaIn);  //FIXME: need a faster version
      const float betaOutUpd =
          betaOut +
          SDL::copysignf(
              alpaka::math::asin(
                  acc,
                  alpaka::math::min(
                      acc, sdOut_dr * SDL::k2Rinv1GeVf / alpaka::math::abs(acc, pt_betaIn), SDL::sinAlphaMax)),
              betaIn);  //FIXME: need a faster version
      betaAv = (alpaka::math::abs(acc, betaOut) > 0.2f * alpaka::math::abs(acc, betaIn))
                   ? (0.5f * (betaInUpd + betaOutUpd))
                   : betaInUpd;

      //1st update
      pt_beta = dr * SDL::k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);  //get a better pt estimate
      betaIn += SDL::copysignf(
          alpaka::math::asin(
              acc,
              alpaka::math::min(acc, sdIn_dr * SDL::k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), SDL::sinAlphaMax)),
          betaIn);  //FIXME: need a faster version
      betaOut += SDL::copysignf(
          alpaka::math::asin(
              acc,
              alpaka::math::min(acc, sdOut_dr * SDL::k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), SDL::sinAlphaMax)),
          betaIn);  //FIXME: need a faster version
      //update the av and pt
      betaAv = 0.5f * (betaIn + betaOut);
      //2nd update
      pt_beta = dr * SDL::k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);  //get a better pt estimate
    }
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runTripletDefaultAlgoPPBB(TAcc const& acc,
                                                                struct SDL::modules& modulesInGPU,
                                                                struct SDL::objectRanges& rangesInGPU,
                                                                struct SDL::miniDoublets& mdsInGPU,
                                                                struct SDL::segments& segmentsInGPU,
                                                                uint16_t& pixelModuleIndex,
                                                                uint16_t& outerInnerLowerModuleIndex,
                                                                uint16_t& outerOuterLowerModuleIndex,
                                                                unsigned int& innerSegmentIndex,
                                                                unsigned int& outerSegmentIndex,
                                                                unsigned int& firstMDIndex,
                                                                unsigned int& secondMDIndex,
                                                                unsigned int thirdMDIndex,
                                                                unsigned int& fourthMDIndex,
                                                                float& /*z_OutLo*/,
                                                                float& /*rt_OutLo*/,
                                                                float& dPhiPos,
                                                                float& dPhi,
                                                                float& betaIn,
                                                                float& betaOut,
                                                                float& pt_beta,
                                                                float& zLo,
                                                                float& zHi,
                                                                float& zLoPointed,
                                                                float& zHiPointed,
                                                                float& sdlCut,
                                                                float& betaOutCut,
                                                                float& deltaBetaCut)  // pixel to BB and BE segments
  {
    bool pass = true;

    bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS);

    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InUp = mdsInGPU.anchorRt[secondMDIndex];
    float rt_OutLo = mdsInGPU.anchorRt[thirdMDIndex];
    float rt_OutUp = mdsInGPU.anchorRt[fourthMDIndex];

    float z_InUp = mdsInGPU.anchorZ[secondMDIndex];
    float z_OutLo = mdsInGPU.anchorZ[thirdMDIndex];

    float x_InLo = mdsInGPU.anchorX[firstMDIndex];
    float x_InUp = mdsInGPU.anchorX[secondMDIndex];
    float x_OutLo = mdsInGPU.anchorX[thirdMDIndex];
    float x_OutUp = mdsInGPU.anchorX[fourthMDIndex];

    float y_InLo = mdsInGPU.anchorY[firstMDIndex];
    float y_InUp = mdsInGPU.anchorY[secondMDIndex];
    float y_OutLo = mdsInGPU.anchorY[thirdMDIndex];
    float y_OutUp = mdsInGPU.anchorY[fourthMDIndex];

    float rt_InOut = rt_InUp;

    pass =
        pass and (alpaka::math::abs(acc, SDL::deltaPhi(acc, x_InUp, y_InUp, x_OutLo, y_OutLo)) <= 0.5f * float(M_PI));
    if (not pass)
      return pass;

    unsigned int pixelSegmentArrayIndex = innerSegmentIndex - rangesInGPU.segmentModuleIndices[pixelModuleIndex];
    float ptIn = segmentsInGPU.ptIn[pixelSegmentArrayIndex];
    float ptSLo = ptIn;
    float px = segmentsInGPU.px[pixelSegmentArrayIndex];
    float py = segmentsInGPU.py[pixelSegmentArrayIndex];
    float pz = segmentsInGPU.pz[pixelSegmentArrayIndex];
    float ptErr = segmentsInGPU.ptErr[pixelSegmentArrayIndex];
    float etaErr = segmentsInGPU.etaErr[pixelSegmentArrayIndex];
    ptSLo = alpaka::math::max(acc, ptCut, ptSLo - 10.0f * alpaka::math::max(acc, ptErr, 0.005f * ptSLo));
    ptSLo = alpaka::math::min(acc, 10.0f, ptSLo);

    float alpha1GeV_OutLo =
        alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / ptCut, sinAlphaMax));
    const float rtRatio_OutLoInOut =
        rt_OutLo / rt_InOut;  // Outer segment beginning rt divided by inner segment beginning rt;

    float dzDrtScale =
        alpaka::math::tan(acc, alpha1GeV_OutLo) / alpha1GeV_OutLo;  // The track can bend in r-z plane slightly
    const float zpitch_InLo = 0.05f;
    const float zpitch_InOut = 0.05f;
    float zpitch_OutLo = (isPS_OutLo ? pixelPSZpitch : strip2SZpitch);
    float zGeom = zpitch_InLo + zpitch_OutLo;
    zHi = z_InUp + (z_InUp + deltaZLum) * (rtRatio_OutLoInOut - 1.f) * (z_InUp < 0.f ? 1.f : dzDrtScale) +
          (zpitch_InOut + zpitch_OutLo);
    zLo = z_InUp + (z_InUp - deltaZLum) * (rtRatio_OutLoInOut - 1.f) * (z_InUp > 0.f ? 1.f : dzDrtScale) -
          (zpitch_InOut + zpitch_OutLo);  //slope-correction only on outer end

    pass = pass and ((z_OutLo >= zLo) && (z_OutLo <= zHi));
    if (not pass)
      return pass;

    const float coshEta = alpaka::math::sqrt(acc, ptIn * ptIn + pz * pz) / ptIn;

    const float drt_OutLo_InUp = (rt_OutLo - rt_InUp);

    const float r3_InUp = alpaka::math::sqrt(acc, z_InUp * z_InUp + rt_InUp * rt_InUp);

    float drt_InSeg = rt_InOut - rt_InLo;

    const float sdlThetaMulsF = 0.015f * alpaka::math::sqrt(acc, 0.1f + 0.2f * (rt_OutLo - rt_InUp) / 50.f) *
                                alpaka::math::sqrt(acc, r3_InUp / rt_InUp);
    const float sdlMuls = sdlThetaMulsF * 3.f / ptCut * 4.f;  // will need a better guess than x4?

    float dzErr = drt_OutLo_InUp * etaErr * coshEta;  //FIXME: check with the calc in the endcap
    dzErr *= dzErr;
    dzErr += 0.03f * 0.03f;  // pixel size x2. ... random for now
    dzErr *= 9.f;            //3 sigma
    dzErr += sdlMuls * sdlMuls * drt_OutLo_InUp * drt_OutLo_InUp / 3.f * coshEta * coshEta;  //sloppy
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

    pass = pass and ((z_OutLo >= zLoPointed) && (z_OutLo <= zHiPointed));
    if (not pass)
      return pass;

    const float sdlPVoff = 0.1f / rt_OutLo;
    sdlCut = alpha1GeV_OutLo + alpaka::math::sqrt(acc, sdlMuls * sdlMuls + sdlPVoff * sdlPVoff);

#ifdef CUT_VALUE_DEBUG
    dPhiPos = SDL::deltaPhi(acc, x_InUp, y_InUp, x_OutUp, y_OutUp);
#endif

    //no dphipos cut
    float midPointX = 0.5f * (x_InLo + x_OutLo);
    float midPointY = 0.5f * (y_InLo + y_OutLo);

    float diffX = x_OutLo - x_InLo;
    float diffY = y_OutLo - y_InLo;

    dPhi = SDL::deltaPhi(acc, midPointX, midPointY, diffX, diffY);

    pass = pass and (alpaka::math::abs(acc, dPhi) <= sdlCut);
    if (not pass)
      return pass;

    //lots of array accesses below this...

    float alpha_InLo = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
    float alpha_OutLo = __H2F(segmentsInGPU.dPhiChanges[outerSegmentIndex]);

    bool isEC_lastLayer = modulesInGPU.subdets[outerOuterLowerModuleIndex] == SDL::Endcap and
                          modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::TwoS;

    float alpha_OutUp, alpha_OutUp_highEdge, alpha_OutUp_lowEdge;
    alpha_OutUp = SDL::deltaPhi(acc, x_OutUp, y_OutUp, x_OutUp - x_OutLo, y_OutUp - y_OutLo);

    alpha_OutUp_highEdge = alpha_OutUp;
    alpha_OutUp_lowEdge = alpha_OutUp;

    float tl_axis_x = x_OutUp - x_InUp;
    float tl_axis_y = y_OutUp - y_InUp;

    float tl_axis_highEdge_x = tl_axis_x;
    float tl_axis_highEdge_y = tl_axis_y;

    float tl_axis_lowEdge_x = tl_axis_x;
    float tl_axis_lowEdge_y = tl_axis_y;

    betaIn = -SDL::deltaPhi(acc, px, py, tl_axis_x, tl_axis_y);
    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;

    betaOut = -alpha_OutUp + SDL::deltaPhi(acc, x_OutUp, y_OutUp, tl_axis_x, tl_axis_y);

    float betaOutRHmin = betaOut;
    float betaOutRHmax = betaOut;

    if (isEC_lastLayer) {
      alpha_OutUp_highEdge = SDL::deltaPhi(acc,
                                           mdsInGPU.anchorHighEdgeX[fourthMDIndex],
                                           mdsInGPU.anchorHighEdgeY[fourthMDIndex],
                                           mdsInGPU.anchorHighEdgeX[fourthMDIndex] - x_OutLo,
                                           mdsInGPU.anchorHighEdgeY[fourthMDIndex] - y_OutLo);
      alpha_OutUp_lowEdge = SDL::deltaPhi(acc,
                                          mdsInGPU.anchorLowEdgeX[fourthMDIndex],
                                          mdsInGPU.anchorLowEdgeY[fourthMDIndex],
                                          mdsInGPU.anchorLowEdgeX[fourthMDIndex] - x_OutLo,
                                          mdsInGPU.anchorLowEdgeY[fourthMDIndex] - y_OutLo);

      tl_axis_highEdge_x = mdsInGPU.anchorHighEdgeX[fourthMDIndex] - x_InUp;
      tl_axis_highEdge_y = mdsInGPU.anchorHighEdgeY[fourthMDIndex] - y_InUp;
      tl_axis_lowEdge_x = mdsInGPU.anchorLowEdgeX[fourthMDIndex] - x_InUp;
      tl_axis_lowEdge_y = mdsInGPU.anchorLowEdgeY[fourthMDIndex] - y_InUp;

      betaOutRHmin = -alpha_OutUp_highEdge + SDL::deltaPhi(acc,
                                                           mdsInGPU.anchorHighEdgeX[fourthMDIndex],
                                                           mdsInGPU.anchorHighEdgeY[fourthMDIndex],
                                                           tl_axis_highEdge_x,
                                                           tl_axis_highEdge_y);
      betaOutRHmax = -alpha_OutUp_lowEdge + SDL::deltaPhi(acc,
                                                          mdsInGPU.anchorLowEdgeX[fourthMDIndex],
                                                          mdsInGPU.anchorLowEdgeY[fourthMDIndex],
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

    runDeltaBetaIterationspT3(acc, betaIn, betaOut, betaAv, pt_beta, rt_InSeg, sdOut_dr, drt_tl_axis, lIn);

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

    const float dBetaMuls =
        sdlThetaMulsF * 4.f /
        alpaka::math::min(
            acc, alpaka::math::abs(acc, pt_beta), SDL::pt_betaMax);  //need to confirm the range-out value of 7 GeV
    const float alphaInAbsReg =
        alpaka::math::max(acc,
                          alpaka::math::abs(acc, alpha_InLo),
                          alpaka::math::asin(acc, alpaka::math::min(acc, rt_InUp * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float alphaOutAbsReg =
        alpaka::math::max(acc,
                          alpaka::math::abs(acc, alpha_OutLo),
                          alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : alpaka::math::abs(acc, alphaInAbsReg * deltaZLum / z_InUp);
    const float dBetaOutLum = lOut < 11 ? 0.0f : alpaka::math::abs(acc, alphaOutAbsReg * deltaZLum / z_OutLo);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);

    const float sinDPhi = alpaka::math::sin(acc, dPhi);
    const float dBetaRIn2 = 0;  // TODO-RH

    float dBetaROut = 0;
    if (isEC_lastLayer) {
      dBetaROut =
          (alpaka::math::sqrt(acc,
                              mdsInGPU.anchorHighEdgeX[fourthMDIndex] * mdsInGPU.anchorHighEdgeX[fourthMDIndex] +
                                  mdsInGPU.anchorHighEdgeY[fourthMDIndex] * mdsInGPU.anchorHighEdgeY[fourthMDIndex]) -
           alpaka::math::sqrt(acc,
                              mdsInGPU.anchorLowEdgeX[fourthMDIndex] * mdsInGPU.anchorLowEdgeX[fourthMDIndex] +
                                  mdsInGPU.anchorLowEdgeY[fourthMDIndex] * mdsInGPU.anchorLowEdgeY[fourthMDIndex])) *
          sinDPhi / drt_tl_axis;
    }

    const float dBetaROut2 = dBetaROut * dBetaROut;

    //FIXME: need faster version
    betaOutCut = alpaka::math::asin(acc, alpaka::math::min(acc, drt_tl_axis * k2Rinv1GeVf / ptCut, sinAlphaMax)) +
                 (0.02f / sdOut_d) + alpaka::math::sqrt(acc, dBetaLum2 + dBetaMuls * dBetaMuls);

    //Cut #6: The real beta cut
    pass = pass and (alpaka::math::abs(acc, betaOut) < betaOutCut);
    if (not pass)
      return pass;
    const float dBetaRes = 0.02f / alpaka::math::min(acc, sdOut_d, drt_InSeg);
    const float dBetaCut2 =
        (dBetaRes * dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2 +
         0.25f *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));
    float dBeta = betaIn - betaOut;

#ifdef CUT_VALUE_DEBUG
    deltaBetaCut = alpaka::math::sqrt(acc, dBetaCut2);
#endif
    pass = pass and (dBeta * dBeta <= dBetaCut2);

    return pass;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runTripletDefaultAlgoPPEE(TAcc const& acc,
                                                                struct SDL::modules& modulesInGPU,
                                                                struct SDL::objectRanges& rangesInGPU,
                                                                struct SDL::miniDoublets& mdsInGPU,
                                                                struct SDL::segments& segmentsInGPU,
                                                                uint16_t& pixelModuleIndex,
                                                                uint16_t& outerInnerLowerModuleIndex,
                                                                uint16_t& outerOuterLowerModuleIndex,
                                                                unsigned int& innerSegmentIndex,
                                                                unsigned int& outerSegmentIndex,
                                                                unsigned int& firstMDIndex,
                                                                unsigned int& secondMDIndex,
                                                                unsigned int& thirdMDIndex,
                                                                unsigned int& fourthMDIndex,
                                                                float& /*z_OutLo*/,
                                                                float& /*rt_OutLo*/,
                                                                float& deltaPhiPos,
                                                                float& dPhi,
                                                                float& betaIn,
                                                                float& betaOut,
                                                                float& pt_beta,
                                                                float& zLo,
                                                                float& rtLo,
                                                                float& rtHi,
                                                                float& sdlCut,
                                                                float& betaInCut,
                                                                float& betaOutCut,
                                                                float& deltaBetaCut,
                                                                float& kZ)  // pixel to EE segments
  {
    bool pass = true;
    bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS);

    float z_InUp = mdsInGPU.anchorZ[secondMDIndex];
    float z_OutLo = mdsInGPU.anchorZ[thirdMDIndex];

    pass = pass and (z_InUp * z_OutLo > 0);
    if (not pass)
      return pass;

    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InUp = mdsInGPU.anchorRt[secondMDIndex];
    float rt_OutLo = mdsInGPU.anchorRt[thirdMDIndex];
    float rt_OutUp = mdsInGPU.anchorRt[fourthMDIndex];

    float x_InLo = mdsInGPU.anchorX[firstMDIndex];
    float x_InUp = mdsInGPU.anchorX[secondMDIndex];
    float x_OutLo = mdsInGPU.anchorX[thirdMDIndex];
    float x_OutUp = mdsInGPU.anchorX[fourthMDIndex];

    float y_InLo = mdsInGPU.anchorY[firstMDIndex];
    float y_InUp = mdsInGPU.anchorY[secondMDIndex];
    float y_OutLo = mdsInGPU.anchorY[thirdMDIndex];
    float y_OutUp = mdsInGPU.anchorY[fourthMDIndex];

    unsigned int pixelSegmentArrayIndex = innerSegmentIndex - rangesInGPU.segmentModuleIndices[pixelModuleIndex];

    float ptIn = segmentsInGPU.ptIn[pixelSegmentArrayIndex];
    float ptSLo = ptIn;
    float px = segmentsInGPU.px[pixelSegmentArrayIndex];
    float py = segmentsInGPU.py[pixelSegmentArrayIndex];
    float pz = segmentsInGPU.pz[pixelSegmentArrayIndex];
    float ptErr = segmentsInGPU.ptErr[pixelSegmentArrayIndex];
    float etaErr = segmentsInGPU.etaErr[pixelSegmentArrayIndex];

    ptSLo = alpaka::math::max(acc, ptCut, ptSLo - 10.0f * alpaka::math::max(acc, ptErr, 0.005f * ptSLo));
    ptSLo = alpaka::math::min(acc, 10.0f, ptSLo);

    float rtOut_o_rtIn = rt_OutLo / rt_InUp;
    const float zpitch_InLo = 0.05f;
    float zpitch_OutLo = (isPS_OutLo ? pixelPSZpitch : strip2SZpitch);
    float zGeom = zpitch_InLo + zpitch_OutLo;

    const float sdlSlope = alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / ptCut, sinAlphaMax));
    const float dzDrtScale = alpaka::math::tan(acc, sdlSlope) / sdlSlope;  //FIXME: need approximate value
    zLo = z_InUp + (z_InUp - deltaZLum) * (rtOut_o_rtIn - 1.f) * (z_InUp > 0.f ? 1.f : dzDrtScale) -
          zGeom;  //slope-correction only on outer end

    const float dLum = SDL::copysignf(deltaZLum, z_InUp);
    bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::PS;

    const float rtGeom1 = isOutSgInnerMDPS
                              ? pixelPSZpitch
                              : strip2SZpitch;           //FIXME: make this chosen by configuration for lay11,12 full PS
    const float zGeom1 = SDL::copysignf(zGeom, z_InUp);  //used in B-E region
    rtLo = rt_InUp * (1.f + (z_OutLo - z_InUp - zGeom1) / (z_InUp + zGeom1 + dLum) / dzDrtScale) -
           rtGeom1;  //slope correction only on the lower end

    float zInForHi = z_InUp - zGeom1 - dLum;
    if (zInForHi * z_InUp < 0)
      zInForHi = SDL::copysignf(0.1f, z_InUp);
    rtHi = rt_InUp * (1.f + (z_OutLo - z_InUp + zGeom1) / zInForHi) + rtGeom1;

    // Cut #2: rt condition
    pass = pass and ((rt_OutLo >= rtLo) && (rt_OutLo <= rtHi));
    if (not pass)
      return pass;

    const float dzOutInAbs = alpaka::math::abs(acc, z_OutLo - z_InUp);
    const float coshEta = alpaka::math::sqrt(acc, ptIn * ptIn + pz * pz) / ptIn;
    const float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);
    const float r3_InUp = alpaka::math::sqrt(acc, z_InUp * z_InUp + rt_InUp * rt_InUp);
    const float sdlThetaMulsF = 0.015f * alpaka::math::sqrt(acc, 0.1f + 0.2f * (rt_OutLo - rt_InUp) / 50.f) *
                                alpaka::math::sqrt(acc, r3_InUp / rt_InUp);
    const float sdlMuls = sdlThetaMulsF * 3.f / ptCut * 4.f;  // will need a better guess than x4?

    float drtErr = etaErr * multDzDr;
    drtErr *= drtErr;
    drtErr += 0.03f * 0.03f;  // pixel size x2. ... random for now
    drtErr *= 9.f;            //3 sigma
    drtErr +=
        sdlMuls * sdlMuls * multDzDr * multDzDr / 3.f * coshEta * coshEta;  //sloppy: relative muls is 1/3 of total muls
    drtErr = alpaka::math::sqrt(acc, drtErr);
    const float drtDzIn = alpaka::math::abs(acc, ptIn / pz);  //all tracks are out-going in endcaps?

    const float drt_OutLo_InUp = (rt_OutLo - rt_InUp);  // drOutIn

    const float rtWindow = drtErr + rtGeom1;
    const float drtMean = drtDzIn * dzOutInAbs *
                          (1.f - drt_OutLo_InUp * drt_OutLo_InUp * 4 * k2Rinv1GeVf * k2Rinv1GeVf / ptIn / ptIn /
                                     24.f);  // with curved path correction
    const float rtLo_point = rt_InUp + drtMean - rtWindow;
    const float rtHi_point = rt_InUp + drtMean + rtWindow;

    // Cut #3: rt-z pointed
    pass = pass and ((rt_OutLo >= rtLo_point) && (rt_OutLo <= rtHi_point));
    if (not pass)
      return pass;

    const float alpha1GeV_OutLo =
        alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / ptCut, sinAlphaMax));
    const float sdlPVoff = 0.1f / rt_OutLo;
    sdlCut = alpha1GeV_OutLo + alpaka::math::sqrt(acc, sdlMuls * sdlMuls + sdlPVoff * sdlPVoff);

    deltaPhiPos = SDL::deltaPhi(acc, x_InUp, y_InUp, x_OutUp, y_OutUp);

    float midPointX = 0.5f * (x_InLo + x_OutLo);
    float midPointY = 0.5f * (y_InLo + y_OutLo);

    float diffX = x_OutLo - x_InLo;
    float diffY = y_OutLo - y_InLo;

    dPhi = SDL::deltaPhi(acc, midPointX, midPointY, diffX, diffY);

    // Cut #5: deltaPhiChange
    pass = pass and (alpaka::math::abs(acc, dPhi) <= sdlCut);
    if (not pass)
      return pass;

    float alpha_InLo = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
    float alpha_OutLo = __H2F(segmentsInGPU.dPhiChanges[outerSegmentIndex]);

    bool isEC_lastLayer = modulesInGPU.subdets[outerOuterLowerModuleIndex] == SDL::Endcap and
                          modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::TwoS;

    float alpha_OutUp, alpha_OutUp_highEdge, alpha_OutUp_lowEdge;

    alpha_OutUp = SDL::deltaPhi(acc, x_OutUp, y_OutUp, x_OutUp - x_OutLo, y_OutUp - y_OutLo);
    alpha_OutUp_highEdge = alpha_OutUp;
    alpha_OutUp_lowEdge = alpha_OutUp;

    float tl_axis_x = x_OutUp - x_InUp;
    float tl_axis_y = y_OutUp - y_InUp;

    float tl_axis_highEdge_x = tl_axis_x;
    float tl_axis_highEdge_y = tl_axis_y;

    float tl_axis_lowEdge_x = tl_axis_x;
    float tl_axis_lowEdge_y = tl_axis_y;

    betaIn = -SDL::deltaPhi(acc, px, py, tl_axis_x, tl_axis_y);
    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;

    betaOut = -alpha_OutUp + SDL::deltaPhi(acc, x_OutUp, y_OutUp, tl_axis_x, tl_axis_y);
    float betaOutRHmin = betaOut;
    float betaOutRHmax = betaOut;

    if (isEC_lastLayer) {
      alpha_OutUp_highEdge = SDL::deltaPhi(acc,
                                           mdsInGPU.anchorHighEdgeX[fourthMDIndex],
                                           mdsInGPU.anchorHighEdgeY[fourthMDIndex],
                                           mdsInGPU.anchorHighEdgeX[fourthMDIndex] - x_OutLo,
                                           mdsInGPU.anchorHighEdgeY[fourthMDIndex] - y_OutLo);
      alpha_OutUp_lowEdge = SDL::deltaPhi(acc,
                                          mdsInGPU.anchorLowEdgeX[fourthMDIndex],
                                          mdsInGPU.anchorLowEdgeY[fourthMDIndex],
                                          mdsInGPU.anchorLowEdgeX[fourthMDIndex] - x_OutLo,
                                          mdsInGPU.anchorLowEdgeY[fourthMDIndex] - y_OutLo);

      tl_axis_highEdge_x = mdsInGPU.anchorHighEdgeX[fourthMDIndex] - x_InUp;
      tl_axis_highEdge_y = mdsInGPU.anchorHighEdgeY[fourthMDIndex] - y_InUp;
      tl_axis_lowEdge_x = mdsInGPU.anchorLowEdgeX[fourthMDIndex] - x_InUp;
      tl_axis_lowEdge_y = mdsInGPU.anchorLowEdgeY[fourthMDIndex] - y_InUp;

      betaOutRHmin = -alpha_OutUp_highEdge + SDL::deltaPhi(acc,
                                                           mdsInGPU.anchorHighEdgeX[fourthMDIndex],
                                                           mdsInGPU.anchorHighEdgeY[fourthMDIndex],
                                                           tl_axis_highEdge_x,
                                                           tl_axis_highEdge_y);
      betaOutRHmax = -alpha_OutUp_lowEdge + SDL::deltaPhi(acc,
                                                          mdsInGPU.anchorLowEdgeX[fourthMDIndex],
                                                          mdsInGPU.anchorLowEdgeY[fourthMDIndex],
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

    runDeltaBetaIterationspT3(acc, betaIn, betaOut, betaAv, pt_beta, rt_InSeg, sdOut_dr, drt_tl_axis, lIn);

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

    const float dBetaMuls =
        sdlThetaMulsF * 4.f /
        alpaka::math::min(
            acc, alpaka::math::abs(acc, pt_beta), SDL::pt_betaMax);  //need to confirm the range-out value of 7 GeV

    const float alphaInAbsReg =
        alpaka::math::max(acc,
                          alpaka::math::abs(acc, alpha_InLo),
                          alpaka::math::asin(acc, alpaka::math::min(acc, rt_InUp * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float alphaOutAbsReg =
        alpaka::math::max(acc,
                          alpaka::math::abs(acc, alpha_OutLo),
                          alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / 3.0f, sinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : alpaka::math::abs(acc, alphaInAbsReg * deltaZLum / z_InUp);
    const float dBetaOutLum = lOut < 11 ? 0.0f : alpaka::math::abs(acc, alphaOutAbsReg * deltaZLum / z_OutLo);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);

    const float sinDPhi = alpaka::math::sin(acc, dPhi);
    const float dBetaRIn2 = 0;  // TODO-RH

    float dBetaROut = 0;
    if (isEC_lastLayer) {
      dBetaROut =
          (alpaka::math::sqrt(acc,
                              mdsInGPU.anchorHighEdgeX[fourthMDIndex] * mdsInGPU.anchorHighEdgeX[fourthMDIndex] +
                                  mdsInGPU.anchorHighEdgeY[fourthMDIndex] * mdsInGPU.anchorHighEdgeY[fourthMDIndex]) -
           alpaka::math::sqrt(acc,
                              mdsInGPU.anchorLowEdgeX[fourthMDIndex] * mdsInGPU.anchorLowEdgeX[fourthMDIndex] +
                                  mdsInGPU.anchorLowEdgeY[fourthMDIndex] * mdsInGPU.anchorLowEdgeY[fourthMDIndex])) *
          sinDPhi / drt_tl_axis;
    }

    const float dBetaROut2 = dBetaROut * dBetaROut;

    betaOutCut =
        alpaka::math::asin(
            acc, alpaka::math::min(acc, drt_tl_axis * k2Rinv1GeVf / ptCut, sinAlphaMax))  //FIXME: need faster version
        + (0.02f / sdOut_d) + alpaka::math::sqrt(acc, dBetaLum2 + dBetaMuls * dBetaMuls);

    //Cut #6: The real beta cut
    pass = pass and (alpaka::math::abs(acc, betaOut) < betaOutCut);
    if (not pass)
      return pass;

    float drt_InSeg = rt_InUp - rt_InLo;

    const float dBetaRes = 0.02f / alpaka::math::min(acc, sdOut_d, drt_InSeg);
    const float dBetaCut2 =
        (dBetaRes * dBetaRes * 2.0f + dBetaMuls * dBetaMuls + dBetaLum2 + dBetaRIn2 + dBetaROut2 +
         0.25f *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));
    float dBeta = betaIn - betaOut;
#ifdef CUT_VALUE_DEBUG
    deltaBetaCut = alpaka::math::sqrt(acc, dBetaCut2);
#endif
    pass = pass and (dBeta * dBeta <= dBetaCut2);
    return pass;
  };
}  // namespace SDL
#endif

#ifndef PixelQuintuplet_cuh
#define PixelQuintuplet_cuh

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
#include "Triplet.h"
#include "Quintuplet.h"
#include "PixelTriplet.h"

namespace SDL {
  struct pixelQuintuplets {
    unsigned int* pixelIndices;
    unsigned int* T5Indices;
    unsigned int* nPixelQuintuplets;
    unsigned int* totOccupancyPixelQuintuplets;
    bool* isDup;
    FPX* score;
    FPX* eta;
    FPX* phi;
    uint8_t* logicalLayers;
    unsigned int* hitIndices;
    uint16_t* lowerModuleIndices;
    FPX* pixelRadius;
    FPX* quintupletRadius;
    FPX* centerX;
    FPX* centerY;
    float* rzChiSquared;
    float* rPhiChiSquared;
    float* rPhiChiSquaredInwards;

    template <typename TBuff>
    void setData(TBuff& pixelQuintupletsBuffer) {
      pixelIndices = alpaka::getPtrNative(pixelQuintupletsBuffer.pixelIndices_buf);
      T5Indices = alpaka::getPtrNative(pixelQuintupletsBuffer.T5Indices_buf);
      nPixelQuintuplets = alpaka::getPtrNative(pixelQuintupletsBuffer.nPixelQuintuplets_buf);
      totOccupancyPixelQuintuplets = alpaka::getPtrNative(pixelQuintupletsBuffer.totOccupancyPixelQuintuplets_buf);
      isDup = alpaka::getPtrNative(pixelQuintupletsBuffer.isDup_buf);
      score = alpaka::getPtrNative(pixelQuintupletsBuffer.score_buf);
      eta = alpaka::getPtrNative(pixelQuintupletsBuffer.eta_buf);
      phi = alpaka::getPtrNative(pixelQuintupletsBuffer.phi_buf);
      logicalLayers = alpaka::getPtrNative(pixelQuintupletsBuffer.logicalLayers_buf);
      hitIndices = alpaka::getPtrNative(pixelQuintupletsBuffer.hitIndices_buf);
      lowerModuleIndices = alpaka::getPtrNative(pixelQuintupletsBuffer.lowerModuleIndices_buf);
      pixelRadius = alpaka::getPtrNative(pixelQuintupletsBuffer.pixelRadius_buf);
      quintupletRadius = alpaka::getPtrNative(pixelQuintupletsBuffer.quintupletRadius_buf);
      centerX = alpaka::getPtrNative(pixelQuintupletsBuffer.centerX_buf);
      centerY = alpaka::getPtrNative(pixelQuintupletsBuffer.centerY_buf);
      rzChiSquared = alpaka::getPtrNative(pixelQuintupletsBuffer.rzChiSquared_buf);
      rPhiChiSquared = alpaka::getPtrNative(pixelQuintupletsBuffer.rPhiChiSquared_buf);
      rPhiChiSquaredInwards = alpaka::getPtrNative(pixelQuintupletsBuffer.rPhiChiSquaredInwards_buf);
    }
  };

  template <typename TDev>
  struct pixelQuintupletsBuffer : pixelQuintuplets {
    Buf<TDev, unsigned int> pixelIndices_buf;
    Buf<TDev, unsigned int> T5Indices_buf;
    Buf<TDev, unsigned int> nPixelQuintuplets_buf;
    Buf<TDev, unsigned int> totOccupancyPixelQuintuplets_buf;
    Buf<TDev, bool> isDup_buf;
    Buf<TDev, FPX> score_buf;
    Buf<TDev, FPX> eta_buf;
    Buf<TDev, FPX> phi_buf;
    Buf<TDev, uint8_t> logicalLayers_buf;
    Buf<TDev, unsigned int> hitIndices_buf;
    Buf<TDev, uint16_t> lowerModuleIndices_buf;
    Buf<TDev, FPX> pixelRadius_buf;
    Buf<TDev, FPX> quintupletRadius_buf;
    Buf<TDev, FPX> centerX_buf;
    Buf<TDev, FPX> centerY_buf;
    Buf<TDev, float> rzChiSquared_buf;
    Buf<TDev, float> rPhiChiSquared_buf;
    Buf<TDev, float> rPhiChiSquaredInwards_buf;

    template <typename TQueue, typename TDevAcc>
    pixelQuintupletsBuffer(unsigned int maxPixelQuintuplets, TDevAcc const& devAccIn, TQueue& queue)
        : pixelIndices_buf(allocBufWrapper<unsigned int>(devAccIn, maxPixelQuintuplets, queue)),
          T5Indices_buf(allocBufWrapper<unsigned int>(devAccIn, maxPixelQuintuplets, queue)),
          nPixelQuintuplets_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          totOccupancyPixelQuintuplets_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          isDup_buf(allocBufWrapper<bool>(devAccIn, maxPixelQuintuplets, queue)),
          score_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuintuplets, queue)),
          eta_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuintuplets, queue)),
          phi_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuintuplets, queue)),
          logicalLayers_buf(allocBufWrapper<uint8_t>(devAccIn, maxPixelQuintuplets * 7, queue)),
          hitIndices_buf(allocBufWrapper<unsigned int>(devAccIn, maxPixelQuintuplets * 14, queue)),
          lowerModuleIndices_buf(allocBufWrapper<uint16_t>(devAccIn, maxPixelQuintuplets * 7, queue)),
          pixelRadius_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuintuplets, queue)),
          quintupletRadius_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuintuplets, queue)),
          centerX_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuintuplets, queue)),
          centerY_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuintuplets, queue)),
          rzChiSquared_buf(allocBufWrapper<float>(devAccIn, maxPixelQuintuplets, queue)),
          rPhiChiSquared_buf(allocBufWrapper<float>(devAccIn, maxPixelQuintuplets, queue)),
          rPhiChiSquaredInwards_buf(allocBufWrapper<float>(devAccIn, maxPixelQuintuplets, queue)) {
      alpaka::memset(queue, nPixelQuintuplets_buf, 0u);
      alpaka::memset(queue, totOccupancyPixelQuintuplets_buf, 0u);
      alpaka::wait(queue);
    }
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addPixelQuintupletToMemory(struct SDL::modules& modulesInGPU,
                                                                 struct SDL::miniDoublets& mdsInGPU,
                                                                 struct SDL::segments& segmentsInGPU,
                                                                 struct SDL::quintuplets& quintupletsInGPU,
                                                                 struct SDL::pixelQuintuplets& pixelQuintupletsInGPU,
                                                                 unsigned int pixelIndex,
                                                                 unsigned int T5Index,
                                                                 unsigned int pixelQuintupletIndex,
                                                                 float& rzChiSquared,
                                                                 float& rPhiChiSquared,
                                                                 float& rPhiChiSquaredInwards,
                                                                 float score,
                                                                 float eta,
                                                                 float phi,
                                                                 float& pixelRadius,
                                                                 float& quintupletRadius,
                                                                 float& centerX,
                                                                 float& centerY) {
    pixelQuintupletsInGPU.pixelIndices[pixelQuintupletIndex] = pixelIndex;
    pixelQuintupletsInGPU.T5Indices[pixelQuintupletIndex] = T5Index;
    pixelQuintupletsInGPU.isDup[pixelQuintupletIndex] = false;
    pixelQuintupletsInGPU.score[pixelQuintupletIndex] = __F2H(score);
    pixelQuintupletsInGPU.eta[pixelQuintupletIndex] = __F2H(eta);
    pixelQuintupletsInGPU.phi[pixelQuintupletIndex] = __F2H(phi);

    pixelQuintupletsInGPU.pixelRadius[pixelQuintupletIndex] = __F2H(pixelRadius);
    pixelQuintupletsInGPU.quintupletRadius[pixelQuintupletIndex] = __F2H(quintupletRadius);
    pixelQuintupletsInGPU.centerX[pixelQuintupletIndex] = __F2H(centerX);
    pixelQuintupletsInGPU.centerY[pixelQuintupletIndex] = __F2H(centerY);

    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex] = 0;
    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex + 1] = 0;
    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex + 2] = quintupletsInGPU.logicalLayers[T5Index * 5];
    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex + 3] = quintupletsInGPU.logicalLayers[T5Index * 5 + 1];
    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex + 4] = quintupletsInGPU.logicalLayers[T5Index * 5 + 2];
    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex + 5] = quintupletsInGPU.logicalLayers[T5Index * 5 + 3];
    pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex + 6] = quintupletsInGPU.logicalLayers[T5Index * 5 + 4];

    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex] =
        segmentsInGPU.innerLowerModuleIndices[pixelIndex];
    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex + 1] =
        segmentsInGPU.outerLowerModuleIndices[pixelIndex];
    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex + 2] =
        quintupletsInGPU.lowerModuleIndices[T5Index * 5];
    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex + 3] =
        quintupletsInGPU.lowerModuleIndices[T5Index * 5 + 1];
    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex + 4] =
        quintupletsInGPU.lowerModuleIndices[T5Index * 5 + 2];
    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex + 5] =
        quintupletsInGPU.lowerModuleIndices[T5Index * 5 + 3];
    pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex + 6] =
        quintupletsInGPU.lowerModuleIndices[T5Index * 5 + 4];

    unsigned int pixelInnerMD = segmentsInGPU.mdIndices[2 * pixelIndex];
    unsigned int pixelOuterMD = segmentsInGPU.mdIndices[2 * pixelIndex + 1];

    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex] = mdsInGPU.anchorHitIndices[pixelInnerMD];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 1] = mdsInGPU.outerHitIndices[pixelInnerMD];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 2] = mdsInGPU.anchorHitIndices[pixelOuterMD];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 3] = mdsInGPU.outerHitIndices[pixelOuterMD];

    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 4] = quintupletsInGPU.hitIndices[10 * T5Index];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 5] = quintupletsInGPU.hitIndices[10 * T5Index + 1];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 6] = quintupletsInGPU.hitIndices[10 * T5Index + 2];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 7] = quintupletsInGPU.hitIndices[10 * T5Index + 3];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 8] = quintupletsInGPU.hitIndices[10 * T5Index + 4];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 9] = quintupletsInGPU.hitIndices[10 * T5Index + 5];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 10] = quintupletsInGPU.hitIndices[10 * T5Index + 6];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 11] = quintupletsInGPU.hitIndices[10 * T5Index + 7];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 12] = quintupletsInGPU.hitIndices[10 * T5Index + 8];
    pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex + 13] = quintupletsInGPU.hitIndices[10 * T5Index + 9];

    pixelQuintupletsInGPU.rzChiSquared[pixelQuintupletIndex] = rzChiSquared;
    pixelQuintupletsInGPU.rPhiChiSquared[pixelQuintupletIndex] = rPhiChiSquared;
    pixelQuintupletsInGPU.rPhiChiSquaredInwards[pixelQuintupletIndex] = rPhiChiSquaredInwards;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT5RZChiSquaredCuts(struct SDL::modules& modulesInGPU,
                                                              uint16_t& lowerModuleIndex1,
                                                              uint16_t& lowerModuleIndex2,
                                                              uint16_t& lowerModuleIndex3,
                                                              uint16_t& lowerModuleIndex4,
                                                              uint16_t& lowerModuleIndex5,
                                                              float& rzChiSquared) {
    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);
    const int layer4 = modulesInGPU.layers[lowerModuleIndex4] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex4] == SDL::TwoS);
    const int layer5 = modulesInGPU.layers[lowerModuleIndex5] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex5] == SDL::TwoS);

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
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT5RPhiChiSquaredCuts(struct SDL::modules& modulesInGPU,
                                                                uint16_t& lowerModuleIndex1,
                                                                uint16_t& lowerModuleIndex2,
                                                                uint16_t& lowerModuleIndex3,
                                                                uint16_t& lowerModuleIndex4,
                                                                uint16_t& lowerModuleIndex5,
                                                                float rPhiChiSquared) {
    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);
    const int layer4 = modulesInGPU.layers[lowerModuleIndex4] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex4] == SDL::TwoS);
    const int layer5 = modulesInGPU.layers[lowerModuleIndex5] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex5] == SDL::TwoS);

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
  };

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
    float absArctanSlope, angleM, xPrime, yPrime, sigma;
    for (size_t i = 0; i < nPoints; i++) {
      absArctanSlope = ((slopes[i] != SDL::SDL_INF) ? alpaka::math::abs(acc, alpaka::math::atan(acc, slopes[i]))
                                                    : 0.5f * float(M_PI));
      if (xs[i] > 0 and ys[i] > 0) {
        angleM = 0.5f * float(M_PI) - absArctanSlope;
      } else if (xs[i] < 0 and ys[i] > 0) {
        angleM = absArctanSlope + 0.5f * float(M_PI);
      } else if (xs[i] < 0 and ys[i] < 0) {
        angleM = -(absArctanSlope + 0.5f * float(M_PI));
      } else if (xs[i] > 0 and ys[i] < 0) {
        angleM = -(0.5f * float(M_PI) - absArctanSlope);
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
      sigma = 2 * alpaka::math::sqrt(
                      acc, (xPrime * delta1[i]) * (xPrime * delta1[i]) + (yPrime * delta2[i]) * (yPrime * delta2[i]));
      chiSquared += (xs[i] * xs[i] + ys[i] * ys[i] - 2 * g * xs[i] - 2 * f * ys[i] + c) *
                    (xs[i] * xs[i] + ys[i] * ys[i] - 2 * g * xs[i] - 2 * f * ys[i] + c) / (sigma * sigma);
    }
    return chiSquared;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void computeSigmasForRegression_pT5(TAcc const& acc,
                                                                     SDL::modules& modulesInGPU,
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
    float inv1 = 0.01f / 0.009f;
    float inv2 = 0.15f / 0.009f;
    float inv3 = 2.4f / 0.009f;
    for (size_t i = 0; i < nPoints; i++) {
      moduleType = modulesInGPU.moduleType[lowerModuleIndices[i]];
      moduleSubdet = modulesInGPU.subdets[lowerModuleIndices[i]];
      moduleSide = modulesInGPU.sides[lowerModuleIndices[i]];
      const float& drdz = modulesInGPU.drdzs[lowerModuleIndices[i]];
      slopes[i] = modulesInGPU.dxdys[lowerModuleIndices[i]];
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
#ifdef Warnings
      else {
        printf("ERROR!!!!! I SHOULDN'T BE HERE!!!! subdet = %d, type = %d, side = %d\n",
               moduleSubdet,
               moduleType,
               moduleSide);
      }
#endif
    }
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computePT5RPhiChiSquared(TAcc const& acc,
                                                                struct SDL::modules& modulesInGPU,
                                                                uint16_t* lowerModuleIndices,
                                                                float& g,
                                                                float& f,
                                                                float& radius,
                                                                float* xs,
                                                                float* ys) {
    /*
        Compute circle parameters from 3 pixel hits, and then use them to compute the chi squared for the outer hits
        */

    float delta1[5], delta2[5], slopes[5];
    bool isFlat[5];
    float chiSquared = 0;

    computeSigmasForRegression_pT5(acc, modulesInGPU, lowerModuleIndices, delta1, delta2, slopes, isFlat);
    chiSquared = computeChiSquaredpT5(acc, 5, xs, ys, delta1, delta2, slopes, isFlat, g, f, radius);

    return chiSquared;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computePT5RPhiChiSquaredInwards(
      struct SDL::modules& modulesInGPU, float& g, float& f, float& r, float* xPix, float* yPix) {
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
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT5RPhiChiSquaredInwardsCuts(struct SDL::modules& modulesInGPU,
                                                                       uint16_t& lowerModuleIndex1,
                                                                       uint16_t& lowerModuleIndex2,
                                                                       uint16_t& lowerModuleIndex3,
                                                                       uint16_t& lowerModuleIndex4,
                                                                       uint16_t& lowerModuleIndex5,
                                                                       float rPhiChiSquared) {
    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex1] == SDL::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex1] == SDL::TwoS);
    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex2] == SDL::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex2] == SDL::TwoS);
    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex3] == SDL::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex3] == SDL::TwoS);
    const int layer4 = modulesInGPU.layers[lowerModuleIndex4] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex4] == SDL::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex4] == SDL::TwoS);
    const int layer5 = modulesInGPU.layers[lowerModuleIndex5] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex5] == SDL::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex5] == SDL::TwoS);

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
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runPixelQuintupletDefaultAlgo(TAcc const& acc,
                                                                    struct SDL::modules& modulesInGPU,
                                                                    struct SDL::objectRanges& rangesInGPU,
                                                                    struct SDL::miniDoublets& mdsInGPU,
                                                                    struct SDL::segments& segmentsInGPU,
                                                                    struct SDL::triplets& tripletsInGPU,
                                                                    struct SDL::quintuplets& quintupletsInGPU,
                                                                    unsigned int& pixelSegmentIndex,
                                                                    unsigned int& quintupletIndex,
                                                                    float& rzChiSquared,
                                                                    float& rPhiChiSquared,
                                                                    float& rPhiChiSquaredInwards,
                                                                    float& pixelRadius,
                                                                    float& quintupletRadius,
                                                                    float& centerX,
                                                                    float& centerY,
                                                                    unsigned int pixelSegmentArrayIndex) {
    bool pass = true;

    unsigned int T5InnerT3Index = quintupletsInGPU.tripletIndices[2 * quintupletIndex];
    unsigned int T5OuterT3Index = quintupletsInGPU.tripletIndices[2 * quintupletIndex + 1];

    float pixelRadiusTemp, pixelRadiusError, tripletRadius, rPhiChiSquaredTemp, rzChiSquaredTemp,
        rPhiChiSquaredInwardsTemp, centerXTemp, centerYTemp;

    pass = pass and runPixelTripletDefaultAlgo(acc,
                                               modulesInGPU,
                                               rangesInGPU,
                                               mdsInGPU,
                                               segmentsInGPU,
                                               tripletsInGPU,
                                               pixelSegmentIndex,
                                               T5InnerT3Index,
                                               pixelRadiusTemp,
                                               pixelRadiusError,
                                               tripletRadius,
                                               centerXTemp,
                                               centerYTemp,
                                               rzChiSquaredTemp,
                                               rPhiChiSquaredTemp,
                                               rPhiChiSquaredInwardsTemp,
                                               false);
    if (not pass)
      return false;

    unsigned int firstSegmentIndex = tripletsInGPU.segmentIndices[2 * T5InnerT3Index];
    unsigned int secondSegmentIndex = tripletsInGPU.segmentIndices[2 * T5InnerT3Index + 1];
    unsigned int thirdSegmentIndex = tripletsInGPU.segmentIndices[2 * T5OuterT3Index];
    unsigned int fourthSegmentIndex = tripletsInGPU.segmentIndices[2 * T5OuterT3Index + 1];

    unsigned int pixelInnerMDIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex];
    unsigned int pixelOuterMDIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex + 1];
    unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 * firstSegmentIndex];
    unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * secondSegmentIndex];
    unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * secondSegmentIndex + 1];
    unsigned int fourthMDIndex = segmentsInGPU.mdIndices[2 * thirdSegmentIndex + 1];
    unsigned int fifthMDIndex = segmentsInGPU.mdIndices[2 * fourthSegmentIndex + 1];

    uint16_t lowerModuleIndex1 = quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex];
    uint16_t lowerModuleIndex2 = quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 1];
    uint16_t lowerModuleIndex3 = quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 2];
    uint16_t lowerModuleIndex4 = quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 3];
    uint16_t lowerModuleIndex5 = quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex + 4];

    uint16_t lowerModuleIndices[5] = {
        lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5};

    float zPix[2] = {mdsInGPU.anchorZ[pixelInnerMDIndex], mdsInGPU.anchorZ[pixelOuterMDIndex]};
    float rtPix[2] = {mdsInGPU.anchorRt[pixelInnerMDIndex], mdsInGPU.anchorRt[pixelOuterMDIndex]};
    float zs[5] = {mdsInGPU.anchorZ[firstMDIndex],
                   mdsInGPU.anchorZ[secondMDIndex],
                   mdsInGPU.anchorZ[thirdMDIndex],
                   mdsInGPU.anchorZ[fourthMDIndex],
                   mdsInGPU.anchorZ[fifthMDIndex]};
    float rts[5] = {mdsInGPU.anchorRt[firstMDIndex],
                    mdsInGPU.anchorRt[secondMDIndex],
                    mdsInGPU.anchorRt[thirdMDIndex],
                    mdsInGPU.anchorRt[fourthMDIndex],
                    mdsInGPU.anchorRt[fifthMDIndex]};

    rzChiSquared = computePT5RZChiSquared(acc, modulesInGPU, lowerModuleIndices, rtPix, zPix, rts, zs);

    if (/*pixelRadius*/ 0 < 5.0f * kR1GeVf) {  // FIXME: pixelRadius is not defined yet
      pass = pass and passPT5RZChiSquaredCuts(modulesInGPU,
                                              lowerModuleIndex1,
                                              lowerModuleIndex2,
                                              lowerModuleIndex3,
                                              lowerModuleIndex4,
                                              lowerModuleIndex5,
                                              rzChiSquared);
      if (not pass)
        return pass;
    }

    //outer T5
    float xs[5] = {mdsInGPU.anchorX[firstMDIndex],
                   mdsInGPU.anchorX[secondMDIndex],
                   mdsInGPU.anchorX[thirdMDIndex],
                   mdsInGPU.anchorX[fourthMDIndex],
                   mdsInGPU.anchorX[fifthMDIndex]};
    float ys[5] = {mdsInGPU.anchorY[firstMDIndex],
                   mdsInGPU.anchorY[secondMDIndex],
                   mdsInGPU.anchorY[thirdMDIndex],
                   mdsInGPU.anchorY[fourthMDIndex],
                   mdsInGPU.anchorY[fifthMDIndex]};

    //get the appropriate radii and centers
    centerX = segmentsInGPU.circleCenterX[pixelSegmentArrayIndex];
    centerY = segmentsInGPU.circleCenterY[pixelSegmentArrayIndex];
    pixelRadius = segmentsInGPU.circleRadius[pixelSegmentArrayIndex];

    float T5CenterX = quintupletsInGPU.regressionG[quintupletIndex];
    float T5CenterY = quintupletsInGPU.regressionF[quintupletIndex];
    quintupletRadius = quintupletsInGPU.regressionRadius[quintupletIndex];

    rPhiChiSquared =
        computePT5RPhiChiSquared(acc, modulesInGPU, lowerModuleIndices, centerX, centerY, pixelRadius, xs, ys);

    if (pixelRadius < 5.0f * kR1GeVf) {
      pass = pass and passPT5RPhiChiSquaredCuts(modulesInGPU,
                                                lowerModuleIndex1,
                                                lowerModuleIndex2,
                                                lowerModuleIndex3,
                                                lowerModuleIndex4,
                                                lowerModuleIndex5,
                                                rPhiChiSquared);
      if (not pass)
        return pass;
    }

    float xPix[] = {mdsInGPU.anchorX[pixelInnerMDIndex], mdsInGPU.anchorX[pixelOuterMDIndex]};
    float yPix[] = {mdsInGPU.anchorY[pixelInnerMDIndex], mdsInGPU.anchorY[pixelOuterMDIndex]};
    rPhiChiSquaredInwards =
        computePT5RPhiChiSquaredInwards(modulesInGPU, T5CenterX, T5CenterY, quintupletRadius, xPix, yPix);

    if (quintupletsInGPU.regressionRadius[quintupletIndex] < 5.0f * kR1GeVf) {
      pass = pass and passPT5RPhiChiSquaredInwardsCuts(modulesInGPU,
                                                       lowerModuleIndex1,
                                                       lowerModuleIndex2,
                                                       lowerModuleIndex3,
                                                       lowerModuleIndex4,
                                                       lowerModuleIndex5,
                                                       rPhiChiSquaredInwards);
      if (not pass)
        return pass;
    }
    //trusting the T5 regression center to also be a good estimate..
    centerX = (centerX + T5CenterX) / 2;
    centerY = (centerY + T5CenterY) / 2;

    //other cuts will be filled here!
    return pass;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computePT5RZChiSquared(TAcc const& acc,
                                                              struct SDL::modules& modulesInGPU,
                                                              uint16_t* lowerModuleIndices,
                                                              float* rtPix,
                                                              float* zPix,
                                                              float* rts,
                                                              float* zs) {
    //use the two anchor hits of the pixel segment to compute the slope
    //then compute the pseudo chi squared of the five outer hits

    float slope = (zPix[1] - zPix[0]) / (rtPix[1] - rtPix[0]);
    float residual = 0;
    float error = 0;
    //hardcoded array indices!!!
    float RMSE = 0;
    for (size_t i = 0; i < 5; i++) {
      uint16_t& lowerModuleIndex = lowerModuleIndices[i];
      const int moduleType = modulesInGPU.moduleType[lowerModuleIndex];
      const int moduleSide = modulesInGPU.sides[lowerModuleIndex];
      const int moduleSubdet = modulesInGPU.subdets[lowerModuleIndex];

      residual = (moduleSubdet == SDL::Barrel) ? (zs[i] - zPix[0]) - slope * (rts[i] - rtPix[0])
                                               : (rts[i] - rtPix[0]) - (zs[i] - zPix[0]) / slope;
      const float& drdz = modulesInGPU.drdzs[lowerModuleIndex];
      //PS Modules
      if (moduleType == 0) {
        error = 0.15f;
      } else  //2S modules
      {
        error = 5.0f;
      }

      //special dispensation to tilted PS modules!
      if (moduleType == 0 and moduleSubdet == SDL::Barrel and moduleSide != Center) {
        error /= alpaka::math::sqrt(acc, 1.f + drdz * drdz);
      }
      RMSE += (residual * residual) / (error * error);
    }

    RMSE = alpaka::math::sqrt(acc, 0.2f * RMSE);
    return RMSE;
  };

  struct createPixelQuintupletsInGPUFromMapv2 {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  struct SDL::modules modulesInGPU,
                                  struct SDL::miniDoublets mdsInGPU,
                                  struct SDL::segments segmentsInGPU,
                                  struct SDL::triplets tripletsInGPU,
                                  struct SDL::quintuplets quintupletsInGPU,
                                  struct SDL::pixelQuintuplets pixelQuintupletsInGPU,
                                  unsigned int* connectedPixelSize,
                                  unsigned int* connectedPixelIndex,
                                  unsigned int nPixelSegments,
                                  struct SDL::objectRanges rangesInGPU) const {
      auto const globalBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridBlockExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (unsigned int i_pLS = globalThreadIdx[1]; i_pLS < nPixelSegments; i_pLS += gridThreadExtent[1]) {
        auto iLSModule_max = connectedPixelIndex[i_pLS] + connectedPixelSize[i_pLS];
        for (unsigned int iLSModule = connectedPixelIndex[i_pLS] + globalBlockIdx[0]; iLSModule < iLSModule_max;
             iLSModule += gridBlockExtent[0]) {
          //these are actual module indices
          uint16_t quintupletLowerModuleIndex = modulesInGPU.connectedPixels[iLSModule];
          if (quintupletLowerModuleIndex >= *modulesInGPU.nLowerModules)
            continue;
          if (modulesInGPU.moduleType[quintupletLowerModuleIndex] == SDL::TwoS)
            continue;
          uint16_t pixelModuleIndex = *modulesInGPU.nLowerModules;
          if (segmentsInGPU.isDup[i_pLS])
            continue;
          unsigned int nOuterQuintuplets = quintupletsInGPU.nQuintuplets[quintupletLowerModuleIndex];

          if (nOuterQuintuplets == 0)
            continue;

          unsigned int pixelSegmentIndex = rangesInGPU.segmentModuleIndices[pixelModuleIndex] + i_pLS;

          //fetch the quintuplet
          for (unsigned int outerQuintupletArrayIndex = globalThreadIdx[2];
               outerQuintupletArrayIndex < nOuterQuintuplets;
               outerQuintupletArrayIndex += gridThreadExtent[2]) {
            unsigned int quintupletIndex =
                rangesInGPU.quintupletModuleIndices[quintupletLowerModuleIndex] + outerQuintupletArrayIndex;

            if (quintupletsInGPU.isDup[quintupletIndex])
              continue;

            float rzChiSquared, rPhiChiSquared, rPhiChiSquaredInwards, pixelRadius, quintupletRadius, centerX, centerY;

            bool success = runPixelQuintupletDefaultAlgo(acc,
                                                         modulesInGPU,
                                                         rangesInGPU,
                                                         mdsInGPU,
                                                         segmentsInGPU,
                                                         tripletsInGPU,
                                                         quintupletsInGPU,
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
              unsigned int totOccupancyPixelQuintuplets =
                  alpaka::atomicOp<alpaka::AtomicAdd>(acc, pixelQuintupletsInGPU.totOccupancyPixelQuintuplets, 1u);
              if (totOccupancyPixelQuintuplets >= N_MAX_PIXEL_QUINTUPLETS) {
#ifdef Warnings
                printf("Pixel Quintuplet excess alert!\n");
#endif
              } else {
                unsigned int pixelQuintupletIndex =
                    alpaka::atomicOp<alpaka::AtomicAdd>(acc, pixelQuintupletsInGPU.nPixelQuintuplets, 1u);
                float eta = __H2F(quintupletsInGPU.eta[quintupletIndex]);
                float phi = __H2F(quintupletsInGPU.phi[quintupletIndex]);

                addPixelQuintupletToMemory(modulesInGPU,
                                           mdsInGPU,
                                           segmentsInGPU,
                                           quintupletsInGPU,
                                           pixelQuintupletsInGPU,
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

                tripletsInGPU.partOfPT5[quintupletsInGPU.tripletIndices[2 * quintupletIndex]] = true;
                tripletsInGPU.partOfPT5[quintupletsInGPU.tripletIndices[2 * quintupletIndex + 1]] = true;
                segmentsInGPU.partOfPT5[i_pLS] = true;
                quintupletsInGPU.partOfPT5[quintupletIndex] = true;
              }  // tot occupancy
            }    // end success
          }      // end T5
        }        // end iLS
      }          // end i_pLS
    }
  };
}  // namespace SDL
#endif
