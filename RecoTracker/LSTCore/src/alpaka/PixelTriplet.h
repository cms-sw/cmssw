#ifndef RecoTracker_LSTCore_src_alpaka_PixelTriplet_h
#define RecoTracker_LSTCore_src_alpaka_PixelTriplet_h

#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/Module.h"

#include "Triplet.h"
#include "Segment.h"
#include "MiniDoublet.h"
#include "Hit.h"
#include "ObjectRanges.h"
#include "Quintuplet.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  // One pixel segment, one outer tracker triplet!
  struct PixelTriplets {
    unsigned int* pixelSegmentIndices;
    unsigned int* tripletIndices;
    unsigned int* nPixelTriplets;
    unsigned int* totOccupancyPixelTriplets;

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
    void setData(TBuff& buf) {
      pixelSegmentIndices = buf.pixelSegmentIndices_buf.data();
      tripletIndices = buf.tripletIndices_buf.data();
      nPixelTriplets = buf.nPixelTriplets_buf.data();
      totOccupancyPixelTriplets = buf.totOccupancyPixelTriplets_buf.data();
      pixelRadius = buf.pixelRadius_buf.data();
      tripletRadius = buf.tripletRadius_buf.data();
      pt = buf.pt_buf.data();
      eta = buf.eta_buf.data();
      phi = buf.phi_buf.data();
      eta_pix = buf.eta_pix_buf.data();
      phi_pix = buf.phi_pix_buf.data();
      score = buf.score_buf.data();
      isDup = buf.isDup_buf.data();
      partOfPT5 = buf.partOfPT5_buf.data();
      logicalLayers = buf.logicalLayers_buf.data();
      hitIndices = buf.hitIndices_buf.data();
      lowerModuleIndices = buf.lowerModuleIndices_buf.data();
      centerX = buf.centerX_buf.data();
      centerY = buf.centerY_buf.data();
      rPhiChiSquared = buf.rPhiChiSquared_buf.data();
      rPhiChiSquaredInwards = buf.rPhiChiSquaredInwards_buf.data();
      rzChiSquared = buf.rzChiSquared_buf.data();
    }
  };

  template <typename TDev>
  struct PixelTripletsBuffer {
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

    PixelTriplets data_;

    template <typename TQueue, typename TDevAcc>
    PixelTripletsBuffer(unsigned int maxPixelTriplets, TDevAcc const& devAccIn, TQueue& queue)
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
          logicalLayers_buf(allocBufWrapper<uint8_t>(devAccIn, maxPixelTriplets * Params_pT3::kLayers, queue)),
          hitIndices_buf(allocBufWrapper<unsigned int>(devAccIn, maxPixelTriplets * Params_pT3::kHits, queue)),
          lowerModuleIndices_buf(allocBufWrapper<uint16_t>(devAccIn, maxPixelTriplets * Params_pT3::kLayers, queue)),
          centerX_buf(allocBufWrapper<FPX>(devAccIn, maxPixelTriplets, queue)),
          centerY_buf(allocBufWrapper<FPX>(devAccIn, maxPixelTriplets, queue)),
          pixelRadiusError_buf(allocBufWrapper<float>(devAccIn, maxPixelTriplets, queue)),
          rPhiChiSquared_buf(allocBufWrapper<float>(devAccIn, maxPixelTriplets, queue)),
          rPhiChiSquaredInwards_buf(allocBufWrapper<float>(devAccIn, maxPixelTriplets, queue)),
          rzChiSquared_buf(allocBufWrapper<float>(devAccIn, maxPixelTriplets, queue)) {
      alpaka::memset(queue, nPixelTriplets_buf, 0u);
      alpaka::memset(queue, totOccupancyPixelTriplets_buf, 0u);
      alpaka::memset(queue, partOfPT5_buf, false);
    }

    inline PixelTriplets const* data() const { return &data_; }
    inline void setData(PixelTripletsBuffer& buf) { data_.setData(buf); }
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addPixelTripletToMemory(MiniDoublets const& mdsInGPU,
                                                              Segments const& segmentsInGPU,
                                                              Triplets const& tripletsInGPU,
                                                              PixelTriplets& pixelTripletsInGPU,
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
    pixelTripletsInGPU.logicalLayers[Params_pT3::kLayers * pixelTripletIndex] = 0;
    pixelTripletsInGPU.logicalLayers[Params_pT3::kLayers * pixelTripletIndex + 1] = 0;
    pixelTripletsInGPU.logicalLayers[Params_pT3::kLayers * pixelTripletIndex + 2] =
        tripletsInGPU.logicalLayers[tripletIndex * Params_T3::kLayers];
    pixelTripletsInGPU.logicalLayers[Params_pT3::kLayers * pixelTripletIndex + 3] =
        tripletsInGPU.logicalLayers[tripletIndex * Params_T3::kLayers + 1];
    pixelTripletsInGPU.logicalLayers[Params_pT3::kLayers * pixelTripletIndex + 4] =
        tripletsInGPU.logicalLayers[tripletIndex * Params_T3::kLayers + 2];

    pixelTripletsInGPU.lowerModuleIndices[Params_pT3::kLayers * pixelTripletIndex] =
        segmentsInGPU.innerLowerModuleIndices[pixelSegmentIndex];
    pixelTripletsInGPU.lowerModuleIndices[Params_pT3::kLayers * pixelTripletIndex + 1] =
        segmentsInGPU.outerLowerModuleIndices[pixelSegmentIndex];
    pixelTripletsInGPU.lowerModuleIndices[Params_pT3::kLayers * pixelTripletIndex + 2] =
        tripletsInGPU.lowerModuleIndices[Params_T3::kLayers * tripletIndex];
    pixelTripletsInGPU.lowerModuleIndices[Params_pT3::kLayers * pixelTripletIndex + 3] =
        tripletsInGPU.lowerModuleIndices[Params_T3::kLayers * tripletIndex + 1];
    pixelTripletsInGPU.lowerModuleIndices[Params_pT3::kLayers * pixelTripletIndex + 4] =
        tripletsInGPU.lowerModuleIndices[Params_T3::kLayers * tripletIndex + 2];

    unsigned int pixelInnerMD = segmentsInGPU.mdIndices[2 * pixelSegmentIndex];
    unsigned int pixelOuterMD = segmentsInGPU.mdIndices[2 * pixelSegmentIndex + 1];

    pixelTripletsInGPU.hitIndices[Params_pT3::kHits * pixelTripletIndex] = mdsInGPU.anchorHitIndices[pixelInnerMD];
    pixelTripletsInGPU.hitIndices[Params_pT3::kHits * pixelTripletIndex + 1] = mdsInGPU.outerHitIndices[pixelInnerMD];
    pixelTripletsInGPU.hitIndices[Params_pT3::kHits * pixelTripletIndex + 2] = mdsInGPU.anchorHitIndices[pixelOuterMD];
    pixelTripletsInGPU.hitIndices[Params_pT3::kHits * pixelTripletIndex + 3] = mdsInGPU.outerHitIndices[pixelOuterMD];

    pixelTripletsInGPU.hitIndices[Params_pT3::kHits * pixelTripletIndex + 4] =
        tripletsInGPU.hitIndices[Params_T3::kHits * tripletIndex];
    pixelTripletsInGPU.hitIndices[Params_pT3::kHits * pixelTripletIndex + 5] =
        tripletsInGPU.hitIndices[Params_T3::kHits * tripletIndex + 1];
    pixelTripletsInGPU.hitIndices[Params_pT3::kHits * pixelTripletIndex + 6] =
        tripletsInGPU.hitIndices[Params_T3::kHits * tripletIndex + 2];
    pixelTripletsInGPU.hitIndices[Params_pT3::kHits * pixelTripletIndex + 7] =
        tripletsInGPU.hitIndices[Params_T3::kHits * tripletIndex + 3];
    pixelTripletsInGPU.hitIndices[Params_pT3::kHits * pixelTripletIndex + 8] =
        tripletsInGPU.hitIndices[Params_T3::kHits * tripletIndex + 4];
    pixelTripletsInGPU.hitIndices[Params_pT3::kHits * pixelTripletIndex + 9] =
        tripletsInGPU.hitIndices[Params_T3::kHits * tripletIndex + 5];
    pixelTripletsInGPU.rPhiChiSquared[pixelTripletIndex] = rPhiChiSquared;
    pixelTripletsInGPU.rPhiChiSquaredInwards[pixelTripletIndex] = rPhiChiSquaredInwards;
    pixelTripletsInGPU.rzChiSquared[pixelTripletIndex] = rzChiSquared;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runPixelTrackletDefaultAlgopT3(TAcc const& acc,
                                                                     Modules const& modulesInGPU,
                                                                     ObjectRanges const& rangesInGPU,
                                                                     MiniDoublets const& mdsInGPU,
                                                                     Segments const& segmentsInGPU,
                                                                     uint16_t pixelLowerModuleIndex,
                                                                     uint16_t outerInnerLowerModuleIndex,
                                                                     uint16_t outerOuterLowerModuleIndex,
                                                                     unsigned int innerSegmentIndex,
                                                                     unsigned int outerSegmentIndex) {
    short outerInnerLowerModuleSubdet = modulesInGPU.subdets[outerInnerLowerModuleIndex];
    short outerOuterLowerModuleSubdet = modulesInGPU.subdets[outerOuterLowerModuleIndex];

    unsigned int firstMDIndex = segmentsInGPU.mdIndices[Params_LS::kLayers * innerSegmentIndex];
    unsigned int secondMDIndex = segmentsInGPU.mdIndices[Params_LS::kLayers * innerSegmentIndex + 1];

    unsigned int thirdMDIndex = segmentsInGPU.mdIndices[Params_LS::kLayers * outerSegmentIndex];
    unsigned int fourthMDIndex = segmentsInGPU.mdIndices[Params_LS::kLayers * outerSegmentIndex + 1];

    if (outerInnerLowerModuleSubdet == Barrel and
        (outerOuterLowerModuleSubdet == Barrel or outerOuterLowerModuleSubdet == Endcap)) {
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
                                       fourthMDIndex);
    } else if (outerInnerLowerModuleSubdet == Endcap and outerOuterLowerModuleSubdet == Endcap) {
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
                                       fourthMDIndex);
    }
    return false;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT3RZChiSquaredCuts(Modules const& modulesInGPU,
                                                              uint16_t lowerModuleIndex1,
                                                              uint16_t lowerModuleIndex2,
                                                              uint16_t lowerModuleIndex3,
                                                              float rzChiSquared) {
    const int layer1 =
        modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == Endcap) +
        5 * (modulesInGPU.subdets[lowerModuleIndex1] == Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == TwoS);
    const int layer2 =
        modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == Endcap) +
        5 * (modulesInGPU.subdets[lowerModuleIndex2] == Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == TwoS);
    const int layer3 =
        modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == Endcap) +
        5 * (modulesInGPU.subdets[lowerModuleIndex3] == Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == TwoS);

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
    float absArctanSlope, angleM, xPrime, yPrime, sigma2;
    for (size_t i = 0; i < nPoints; i++) {
      absArctanSlope =
          ((slopes[i] != lst_INF) ? alpaka::math::abs(acc, alpaka::math::atan(acc, slopes[i])) : 0.5f * float(M_PI));
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
      sigma2 = 4 * ((xPrime * delta1[i]) * (xPrime * delta1[i]) + (yPrime * delta2[i]) * (yPrime * delta2[i]));
      chiSquared += (xs[i] * xs[i] + ys[i] * ys[i] - 2 * g * xs[i] - 2 * f * ys[i] + c) *
                    (xs[i] * xs[i] + ys[i] * ys[i] - 2 * g * xs[i] - 2 * f * ys[i] + c) / sigma2;
    }
    return chiSquared;
  };

  //TODO: merge this one and the pT5 function later into a single function
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computePT3RPhiChiSquared(TAcc const& acc,
                                                                Modules const& modulesInGPU,
                                                                uint16_t* lowerModuleIndices,
                                                                float g,
                                                                float f,
                                                                float radius,
                                                                float* xs,
                                                                float* ys) {
    float delta1[3]{}, delta2[3]{}, slopes[3];
    bool isFlat[3]{};
    float chiSquared = 0;
    float inv1 = kWidthPS / kWidth2S;
    float inv2 = kPixelPSZpitch / kWidth2S;
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
#ifdef WARNINGS
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
      float g, float f, float r, float* xPix, float* yPix) {
    float residual = (xPix[0] - g) * (xPix[0] - g) + (yPix[0] - f) * (yPix[0] - f) - r * r;
    float chiSquared = residual * residual;
    residual = (xPix[1] - g) * (xPix[1] - g) + (yPix[1] - f) * (yPix[1] - f) - r * r;
    chiSquared += residual * residual;

    chiSquared *= 0.5f;
    return chiSquared;
  };

  //90pc threshold
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT3RPhiChiSquaredCuts(Modules const& modulesInGPU,
                                                                uint16_t lowerModuleIndex1,
                                                                uint16_t lowerModuleIndex2,
                                                                uint16_t lowerModuleIndex3,
                                                                float chiSquared) {
    const int layer1 =
        modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == Endcap) +
        5 * (modulesInGPU.subdets[lowerModuleIndex1] == Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == TwoS);
    const int layer2 =
        modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == Endcap) +
        5 * (modulesInGPU.subdets[lowerModuleIndex2] == Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == TwoS);
    const int layer3 =
        modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == Endcap) +
        5 * (modulesInGPU.subdets[lowerModuleIndex3] == Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == TwoS);

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

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT3RPhiChiSquaredInwardsCuts(Modules const& modulesInGPU,
                                                                       uint16_t lowerModuleIndex1,
                                                                       uint16_t lowerModuleIndex2,
                                                                       uint16_t lowerModuleIndex3,
                                                                       float chiSquared) {
    const int layer1 =
        modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == Endcap) +
        5 * (modulesInGPU.subdets[lowerModuleIndex1] == Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == TwoS);
    const int layer2 =
        modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == Endcap) +
        5 * (modulesInGPU.subdets[lowerModuleIndex2] == Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == TwoS);
    const int layer3 =
        modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == Endcap) +
        5 * (modulesInGPU.subdets[lowerModuleIndex3] == Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == TwoS);

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

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool checkIntervalOverlappT3(float firstMin,
                                                              float firstMax,
                                                              float secondMin,
                                                              float secondMax) {
    return ((firstMin <= secondMin) && (secondMin < firstMax)) || ((secondMin < firstMin) && (firstMin < secondMax));
  };

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
  };

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
  };

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
  };

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
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passRadiusCriterion(TAcc const& acc,
                                                          Modules const& modulesInGPU,
                                                          float pixelRadius,
                                                          float pixelRadiusError,
                                                          float tripletRadius,
                                                          int16_t lowerModuleIndex,
                                                          uint16_t middleModuleIndex,
                                                          uint16_t upperModuleIndex) {
    if (modulesInGPU.subdets[lowerModuleIndex] == Endcap) {
      return passRadiusCriterionEEE(acc, pixelRadius, pixelRadiusError, tripletRadius);
    } else if (modulesInGPU.subdets[middleModuleIndex] == Endcap) {
      return passRadiusCriterionBEE(acc, pixelRadius, pixelRadiusError, tripletRadius);
    } else if (modulesInGPU.subdets[upperModuleIndex] == Endcap) {
      return passRadiusCriterionBBE(acc, pixelRadius, pixelRadiusError, tripletRadius);
    } else {
      return passRadiusCriterionBBB(acc, pixelRadius, pixelRadiusError, tripletRadius);
    }
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computePT3RZChiSquared(TAcc const& acc,
                                                              Modules const& modulesInGPU,
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
      const int moduleType = modulesInGPU.moduleType[lowerModuleIndex];
      const int moduleSide = modulesInGPU.sides[lowerModuleIndex];
      const int moduleSubdet = modulesInGPU.subdets[lowerModuleIndex];

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
        float drdz = modulesInGPU.drdzs[lowerModuleIndex];
        error2 /= (1 + drdz * drdz);
      }
      RMSE += (residual * residual) / error2;
    }

    RMSE = alpaka::math::sqrt(acc, 0.2f * RMSE);  // Divided by the degree of freedom 5.

    return RMSE;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runPixelTripletDefaultAlgo(TAcc const& acc,
                                                                 Modules const& modulesInGPU,
                                                                 ObjectRanges const& rangesInGPU,
                                                                 MiniDoublets const& mdsInGPU,
                                                                 Segments const& segmentsInGPU,
                                                                 Triplets const& tripletsInGPU,
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
    uint16_t pixelModuleIndex = segmentsInGPU.innerLowerModuleIndices[pixelSegmentIndex];

    uint16_t lowerModuleIndex = tripletsInGPU.lowerModuleIndices[Params_T3::kLayers * tripletIndex];
    uint16_t middleModuleIndex = tripletsInGPU.lowerModuleIndices[Params_T3::kLayers * tripletIndex + 1];
    uint16_t upperModuleIndex = tripletsInGPU.lowerModuleIndices[Params_T3::kLayers * tripletIndex + 2];

    {
      // pixel segment vs inner segment of the triplet
      if (not runPixelTrackletDefaultAlgopT3(acc,
                                             modulesInGPU,
                                             rangesInGPU,
                                             mdsInGPU,
                                             segmentsInGPU,
                                             pixelModuleIndex,
                                             lowerModuleIndex,
                                             middleModuleIndex,
                                             pixelSegmentIndex,
                                             tripletsInGPU.segmentIndices[Params_LS::kLayers * tripletIndex]))
        return false;

      //pixel segment vs outer segment of triplet
      if (not runPixelTrackletDefaultAlgopT3(acc,
                                             modulesInGPU,
                                             rangesInGPU,
                                             mdsInGPU,
                                             segmentsInGPU,
                                             pixelModuleIndex,
                                             middleModuleIndex,
                                             upperModuleIndex,
                                             pixelSegmentIndex,
                                             tripletsInGPU.segmentIndices[Params_LS::kLayers * tripletIndex + 1]))
        return false;
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

    unsigned int pixelInnerMDIndex = segmentsInGPU.mdIndices[Params_pLS::kLayers * pixelSegmentIndex];
    unsigned int pixelOuterMDIndex = segmentsInGPU.mdIndices[Params_pLS::kLayers * pixelSegmentIndex + 1];

    pixelRadius = pixelSegmentPt * kR1GeVf;
    float pixelRadiusError = pixelSegmentPtError * kR1GeVf;
    unsigned int tripletInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * tripletIndex];
    unsigned int tripletOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * tripletIndex + 1];

    unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 * tripletInnerSegmentIndex];
    unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * tripletInnerSegmentIndex + 1];
    unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * tripletOuterSegmentIndex + 1];

    float xs[Params_T3::kLayers] = {
        mdsInGPU.anchorX[firstMDIndex], mdsInGPU.anchorX[secondMDIndex], mdsInGPU.anchorX[thirdMDIndex]};
    float ys[Params_T3::kLayers] = {
        mdsInGPU.anchorY[firstMDIndex], mdsInGPU.anchorY[secondMDIndex], mdsInGPU.anchorY[thirdMDIndex]};

    float g, f;
    tripletRadius = tripletsInGPU.circleRadius[tripletIndex];
    g = tripletsInGPU.circleCenterX[tripletIndex];
    f = tripletsInGPU.circleCenterY[tripletIndex];

    if (not passRadiusCriterion(acc,
                                modulesInGPU,
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
          mdsInGPU.anchorRt[firstMDIndex], mdsInGPU.anchorRt[secondMDIndex], mdsInGPU.anchorRt[thirdMDIndex]};
      float zs[Params_T3::kLayers] = {
          mdsInGPU.anchorZ[firstMDIndex], mdsInGPU.anchorZ[secondMDIndex], mdsInGPU.anchorZ[thirdMDIndex]};
      float rtPix[Params_pLS::kLayers] = {mdsInGPU.anchorRt[pixelInnerMDIndex], mdsInGPU.anchorRt[pixelOuterMDIndex]};
      float xPix[Params_pLS::kLayers] = {mdsInGPU.anchorX[pixelInnerMDIndex], mdsInGPU.anchorX[pixelOuterMDIndex]};
      float yPix[Params_pLS::kLayers] = {mdsInGPU.anchorY[pixelInnerMDIndex], mdsInGPU.anchorY[pixelOuterMDIndex]};
      float zPix[Params_pLS::kLayers] = {mdsInGPU.anchorZ[pixelInnerMDIndex], mdsInGPU.anchorZ[pixelOuterMDIndex]};

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
      if (not passPT3RZChiSquaredCuts(
              modulesInGPU, lowerModuleIndex, middleModuleIndex, upperModuleIndex, rzChiSquared))
        return false;
    } else {
      rzChiSquared = -1;
    }

    rPhiChiSquared =
        computePT3RPhiChiSquared(acc, modulesInGPU, lowerModuleIndices, pixelG, pixelF, pixelRadiusPCA, xs, ys);

    if (runChiSquaredCuts and pixelSegmentPt < 5.0f) {
      if (not passPT3RPhiChiSquaredCuts(
              modulesInGPU, lowerModuleIndex, middleModuleIndex, upperModuleIndex, rPhiChiSquared))
        return false;
    }

    float xPix[Params_pLS::kLayers] = {mdsInGPU.anchorX[pixelInnerMDIndex], mdsInGPU.anchorX[pixelOuterMDIndex]};
    float yPix[Params_pLS::kLayers] = {mdsInGPU.anchorY[pixelInnerMDIndex], mdsInGPU.anchorY[pixelOuterMDIndex]};
    rPhiChiSquaredInwards = computePT3RPhiChiSquaredInwards(g, f, tripletRadius, xPix, yPix);

    if (runChiSquaredCuts and pixelSegmentPt < 5.0f) {
      if (not passPT3RPhiChiSquaredInwardsCuts(
              modulesInGPU, lowerModuleIndex, middleModuleIndex, upperModuleIndex, rPhiChiSquaredInwards))
        return false;
    }
    centerX = 0;
    centerY = 0;
    return true;
  };

  struct CreatePixelTripletsInGPUFromMapv2 {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  Modules modulesInGPU,
                                  ObjectRanges rangesInGPU,
                                  MiniDoublets mdsInGPU,
                                  Segments segmentsInGPU,
                                  Triplets tripletsInGPU,
                                  PixelTriplets pixelTripletsInGPU,
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
#ifdef WARNINGS
          if (tripletLowerModuleIndex >= *modulesInGPU.nLowerModules) {
            printf("tripletLowerModuleIndex %d >= modulesInGPU.nLowerModules %d \n",
                   tripletLowerModuleIndex,
                   *modulesInGPU.nLowerModules);
            continue;  //sanity check
          }
#endif
          //Removes 2S-2S :FIXME: filter these out in the pixel map
          if (modulesInGPU.moduleType[tripletLowerModuleIndex] == TwoS)
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
            if (modulesInGPU.moduleType[tripletsInGPU.lowerModuleIndices[3 * outerTripletIndex + 1]] == TwoS)
              continue;  //REMOVES PS-2S

            if (tripletsInGPU.partOfPT5[outerTripletIndex])
              continue;  //don't create pT3s for T3s accounted in pT5s

            float pixelRadius, tripletRadius, rPhiChiSquared, rzChiSquared, rPhiChiSquaredInwards, centerX, centerY;
            bool success = runPixelTripletDefaultAlgo(acc,
                                                      modulesInGPU,
                                                      rangesInGPU,
                                                      mdsInGPU,
                                                      segmentsInGPU,
                                                      tripletsInGPU,
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
                  mdsInGPU.anchorPhi[segmentsInGPU.mdIndices[2 * tripletsInGPU.segmentIndices[2 * outerTripletIndex] +
                                                             layer2_adjustment]];
              float eta =
                  mdsInGPU.anchorEta[segmentsInGPU.mdIndices[2 * tripletsInGPU.segmentIndices[2 * outerTripletIndex] +
                                                             layer2_adjustment]];
              float eta_pix = segmentsInGPU.eta[i_pLS];
              float phi_pix = segmentsInGPU.phi[i_pLS];
              float pt = segmentsInGPU.ptIn[i_pLS];
              float score = rPhiChiSquared + rPhiChiSquaredInwards;
              unsigned int totOccupancyPixelTriplets = alpaka::atomicAdd(
                  acc, pixelTripletsInGPU.totOccupancyPixelTriplets, 1u, alpaka::hierarchy::Threads{});
              if (totOccupancyPixelTriplets >= n_max_pixel_triplets) {
#ifdef WARNINGS
                printf("Pixel Triplet excess alert!\n");
#endif
              } else {
                unsigned int pixelTripletIndex =
                    alpaka::atomicAdd(acc, pixelTripletsInGPU.nPixelTriplets, 1u, alpaka::hierarchy::Threads{});
                addPixelTripletToMemory(mdsInGPU,
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
        }  // for iLSModule < iLSModule_max
      }  // for i_pLS
    }
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void runDeltaBetaIterationspT3(TAcc const& acc,
                                                                float& betaIn,
                                                                float& betaOut,
                                                                float betaAv,
                                                                float& pt_beta,
                                                                float sdIn_dr,
                                                                float sdOut_dr,
                                                                float dr,
                                                                float lIn) {
    if (lIn == 0) {
      betaOut += alpaka::math::copysign(
          acc,
          alpaka::math::asin(
              acc, alpaka::math::min(acc, sdOut_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), kSinAlphaMax)),
          betaOut);
      return;
    }

    if (betaIn * betaOut > 0.f and
        (alpaka::math::abs(acc, pt_beta) < 4.f * kPt_betaMax or
         (lIn >= 11 and alpaka::math::abs(acc, pt_beta) <
                            8.f * kPt_betaMax)))  //and the pt_beta is well-defined; less strict for endcap-endcap
    {
      const float betaInUpd =
          betaIn +
          alpaka::math::copysign(
              acc,
              alpaka::math::asin(
                  acc, alpaka::math::min(acc, sdIn_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), kSinAlphaMax)),
              betaIn);  //FIXME: need a faster version
      const float betaOutUpd =
          betaOut +
          alpaka::math::copysign(
              acc,
              alpaka::math::asin(
                  acc, alpaka::math::min(acc, sdOut_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), kSinAlphaMax)),
              betaOut);  //FIXME: need a faster version
      betaAv = 0.5f * (betaInUpd + betaOutUpd);

      //1st update
      const float pt_beta_inv =
          1.f / alpaka::math::abs(acc, dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv));  //get a better pt estimate

      betaIn += alpaka::math::copysign(
          acc,
          alpaka::math::asin(acc, alpaka::math::min(acc, sdIn_dr * k2Rinv1GeVf * pt_beta_inv, kSinAlphaMax)),
          betaIn);  //FIXME: need a faster version
      betaOut += alpaka::math::copysign(
          acc,
          alpaka::math::asin(acc, alpaka::math::min(acc, sdOut_dr * k2Rinv1GeVf * pt_beta_inv, kSinAlphaMax)),
          betaOut);  //FIXME: need a faster version
      //update the av and pt
      betaAv = 0.5f * (betaIn + betaOut);
      //2nd update
      pt_beta = dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);  //get a better pt estimate
    } else if (lIn < 11 && alpaka::math::abs(acc, betaOut) < 0.2f * alpaka::math::abs(acc, betaIn) &&
               alpaka::math::abs(acc, pt_beta) < 12.f * kPt_betaMax)  //use betaIn sign as ref
    {
      const float pt_betaIn = dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaIn);

      const float betaInUpd =
          betaIn +
          alpaka::math::copysign(
              acc,
              alpaka::math::asin(
                  acc, alpaka::math::min(acc, sdIn_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_betaIn), kSinAlphaMax)),
              betaIn);  //FIXME: need a faster version
      const float betaOutUpd =
          betaOut +
          alpaka::math::copysign(
              acc,
              alpaka::math::asin(
                  acc,
                  alpaka::math::min(acc, sdOut_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_betaIn), kSinAlphaMax)),
              betaIn);  //FIXME: need a faster version
      betaAv = (alpaka::math::abs(acc, betaOut) > 0.2f * alpaka::math::abs(acc, betaIn))
                   ? (0.5f * (betaInUpd + betaOutUpd))
                   : betaInUpd;

      //1st update
      pt_beta = dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);  //get a better pt estimate
      betaIn += alpaka::math::copysign(
          acc,
          alpaka::math::asin(
              acc, alpaka::math::min(acc, sdIn_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), kSinAlphaMax)),
          betaIn);  //FIXME: need a faster version
      betaOut += alpaka::math::copysign(
          acc,
          alpaka::math::asin(
              acc, alpaka::math::min(acc, sdOut_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), kSinAlphaMax)),
          betaIn);  //FIXME: need a faster version
      //update the av and pt
      betaAv = 0.5f * (betaIn + betaOut);
      //2nd update
      pt_beta = dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);  //get a better pt estimate
    }
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runTripletDefaultAlgoPPBB(TAcc const& acc,
                                                                Modules const& modulesInGPU,
                                                                ObjectRanges const& rangesInGPU,
                                                                MiniDoublets const& mdsInGPU,
                                                                Segments const& segmentsInGPU,
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

    bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == PS);

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

    if (alpaka::math::abs(acc, deltaPhi(acc, x_InUp, y_InUp, x_OutLo, y_OutLo)) > 0.5f * float(M_PI))
      return false;

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

    float alpha_InLo = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
    float alpha_OutLo = __H2F(segmentsInGPU.dPhiChanges[outerSegmentIndex]);

    bool isEC_lastLayer = modulesInGPU.subdets[outerOuterLowerModuleIndex] == Endcap and
                          modulesInGPU.moduleType[outerOuterLowerModuleIndex] == TwoS;

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
                                      mdsInGPU.anchorHighEdgeX[fourthMDIndex],
                                      mdsInGPU.anchorHighEdgeY[fourthMDIndex],
                                      mdsInGPU.anchorHighEdgeX[fourthMDIndex] - x_OutLo,
                                      mdsInGPU.anchorHighEdgeY[fourthMDIndex] - y_OutLo);
      alpha_OutUp_lowEdge = deltaPhi(acc,
                                     mdsInGPU.anchorLowEdgeX[fourthMDIndex],
                                     mdsInGPU.anchorLowEdgeY[fourthMDIndex],
                                     mdsInGPU.anchorLowEdgeX[fourthMDIndex] - x_OutLo,
                                     mdsInGPU.anchorLowEdgeY[fourthMDIndex] - y_OutLo);

      tl_axis_highEdge_x = mdsInGPU.anchorHighEdgeX[fourthMDIndex] - x_InUp;
      tl_axis_highEdge_y = mdsInGPU.anchorHighEdgeY[fourthMDIndex] - y_InUp;
      tl_axis_lowEdge_x = mdsInGPU.anchorLowEdgeX[fourthMDIndex] - x_InUp;
      tl_axis_lowEdge_y = mdsInGPU.anchorLowEdgeY[fourthMDIndex] - y_InUp;

      betaOutRHmin = -alpha_OutUp_highEdge + deltaPhi(acc,
                                                      mdsInGPU.anchorHighEdgeX[fourthMDIndex],
                                                      mdsInGPU.anchorHighEdgeY[fourthMDIndex],
                                                      tl_axis_highEdge_x,
                                                      tl_axis_highEdge_y);
      betaOutRHmax = -alpha_OutUp_lowEdge + deltaPhi(acc,
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
                                                                Modules const& modulesInGPU,
                                                                ObjectRanges const& rangesInGPU,
                                                                MiniDoublets const& mdsInGPU,
                                                                Segments const& segmentsInGPU,
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

    bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == PS);

    float z_InUp = mdsInGPU.anchorZ[secondMDIndex];
    float z_OutLo = mdsInGPU.anchorZ[thirdMDIndex];

    if (z_InUp * z_OutLo <= 0)
      return false;

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

    const float zpitch_InLo = 0.05f;
    float zpitch_OutLo = (isPS_OutLo ? kPixelPSZpitch : kStrip2SZpitch);
    float zGeom = zpitch_InLo + zpitch_OutLo;

    const float slope = alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / ptCut, kSinAlphaMax));
    const float dzDrtScale = alpaka::math::tan(acc, slope) / slope;  //FIXME: need approximate value

    const float dLum = alpaka::math::copysign(acc, kDeltaZLum, z_InUp);
    bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerInnerLowerModuleIndex] == PS;

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

    float alpha_InLo = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
    float alpha_OutLo = __H2F(segmentsInGPU.dPhiChanges[outerSegmentIndex]);

    bool isEC_lastLayer = modulesInGPU.subdets[outerOuterLowerModuleIndex] == Endcap and
                          modulesInGPU.moduleType[outerOuterLowerModuleIndex] == TwoS;

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
                                      mdsInGPU.anchorHighEdgeX[fourthMDIndex],
                                      mdsInGPU.anchorHighEdgeY[fourthMDIndex],
                                      mdsInGPU.anchorHighEdgeX[fourthMDIndex] - x_OutLo,
                                      mdsInGPU.anchorHighEdgeY[fourthMDIndex] - y_OutLo);
      alpha_OutUp_lowEdge = deltaPhi(acc,
                                     mdsInGPU.anchorLowEdgeX[fourthMDIndex],
                                     mdsInGPU.anchorLowEdgeY[fourthMDIndex],
                                     mdsInGPU.anchorLowEdgeX[fourthMDIndex] - x_OutLo,
                                     mdsInGPU.anchorLowEdgeY[fourthMDIndex] - y_OutLo);

      tl_axis_highEdge_x = mdsInGPU.anchorHighEdgeX[fourthMDIndex] - x_InUp;
      tl_axis_highEdge_y = mdsInGPU.anchorHighEdgeY[fourthMDIndex] - y_InUp;
      tl_axis_lowEdge_x = mdsInGPU.anchorLowEdgeX[fourthMDIndex] - x_InUp;
      tl_axis_lowEdge_y = mdsInGPU.anchorLowEdgeY[fourthMDIndex] - y_InUp;

      betaOutRHmin = -alpha_OutUp_highEdge + deltaPhi(acc,
                                                      mdsInGPU.anchorHighEdgeX[fourthMDIndex],
                                                      mdsInGPU.anchorHighEdgeY[fourthMDIndex],
                                                      tl_axis_highEdge_x,
                                                      tl_axis_highEdge_y);
      betaOutRHmax = -alpha_OutUp_lowEdge + deltaPhi(acc,
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
