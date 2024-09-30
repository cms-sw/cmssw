#ifndef RecoTracker_LSTCore_src_alpaka_PixelQuintuplet_h
#define RecoTracker_LSTCore_src_alpaka_PixelQuintuplet_h

#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/Module.h"

#include "Segment.h"
#include "MiniDoublet.h"
#include "Hit.h"
#include "Triplet.h"
#include "Quintuplet.h"
#include "PixelTriplet.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  struct PixelQuintuplets {
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
    void setData(TBuff& buf) {
      pixelIndices = buf.pixelIndices_buf.data();
      T5Indices = buf.T5Indices_buf.data();
      nPixelQuintuplets = buf.nPixelQuintuplets_buf.data();
      totOccupancyPixelQuintuplets = buf.totOccupancyPixelQuintuplets_buf.data();
      isDup = buf.isDup_buf.data();
      score = buf.score_buf.data();
      eta = buf.eta_buf.data();
      phi = buf.phi_buf.data();
      logicalLayers = buf.logicalLayers_buf.data();
      hitIndices = buf.hitIndices_buf.data();
      lowerModuleIndices = buf.lowerModuleIndices_buf.data();
      pixelRadius = buf.pixelRadius_buf.data();
      quintupletRadius = buf.quintupletRadius_buf.data();
      centerX = buf.centerX_buf.data();
      centerY = buf.centerY_buf.data();
      rzChiSquared = buf.rzChiSquared_buf.data();
      rPhiChiSquared = buf.rPhiChiSquared_buf.data();
      rPhiChiSquaredInwards = buf.rPhiChiSquaredInwards_buf.data();
    }
  };

  template <typename TDev>
  struct PixelQuintupletsBuffer {
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

    PixelQuintuplets data_;

    template <typename TQueue, typename TDevAcc>
    PixelQuintupletsBuffer(unsigned int maxPixelQuintuplets, TDevAcc const& devAccIn, TQueue& queue)
        : pixelIndices_buf(allocBufWrapper<unsigned int>(devAccIn, maxPixelQuintuplets, queue)),
          T5Indices_buf(allocBufWrapper<unsigned int>(devAccIn, maxPixelQuintuplets, queue)),
          nPixelQuintuplets_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          totOccupancyPixelQuintuplets_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          isDup_buf(allocBufWrapper<bool>(devAccIn, maxPixelQuintuplets, queue)),
          score_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuintuplets, queue)),
          eta_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuintuplets, queue)),
          phi_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuintuplets, queue)),
          logicalLayers_buf(allocBufWrapper<uint8_t>(devAccIn, maxPixelQuintuplets * Params_pT5::kLayers, queue)),
          hitIndices_buf(allocBufWrapper<unsigned int>(devAccIn, maxPixelQuintuplets * Params_pT5::kHits, queue)),
          lowerModuleIndices_buf(allocBufWrapper<uint16_t>(devAccIn, maxPixelQuintuplets * Params_pT5::kLayers, queue)),
          pixelRadius_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuintuplets, queue)),
          quintupletRadius_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuintuplets, queue)),
          centerX_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuintuplets, queue)),
          centerY_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuintuplets, queue)),
          rzChiSquared_buf(allocBufWrapper<float>(devAccIn, maxPixelQuintuplets, queue)),
          rPhiChiSquared_buf(allocBufWrapper<float>(devAccIn, maxPixelQuintuplets, queue)),
          rPhiChiSquaredInwards_buf(allocBufWrapper<float>(devAccIn, maxPixelQuintuplets, queue)) {
      alpaka::memset(queue, nPixelQuintuplets_buf, 0u);
      alpaka::memset(queue, totOccupancyPixelQuintuplets_buf, 0u);
    }

    inline PixelQuintuplets const* data() const { return &data_; }
    inline void setData(PixelQuintupletsBuffer& buf) { data_.setData(buf); }
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addPixelQuintupletToMemory(Modules const& modulesInGPU,
                                                                 MiniDoublets const& mdsInGPU,
                                                                 Segments const& segmentsInGPU,
                                                                 Quintuplets const& quintupletsInGPU,
                                                                 PixelQuintuplets& pixelQuintupletsInGPU,
                                                                 unsigned int pixelIndex,
                                                                 unsigned int T5Index,
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

    pixelQuintupletsInGPU.logicalLayers[Params_pT5::kLayers * pixelQuintupletIndex] = 0;
    pixelQuintupletsInGPU.logicalLayers[Params_pT5::kLayers * pixelQuintupletIndex + 1] = 0;
    pixelQuintupletsInGPU.logicalLayers[Params_pT5::kLayers * pixelQuintupletIndex + 2] =
        quintupletsInGPU.logicalLayers[T5Index * Params_T5::kLayers];
    pixelQuintupletsInGPU.logicalLayers[Params_pT5::kLayers * pixelQuintupletIndex + 3] =
        quintupletsInGPU.logicalLayers[T5Index * Params_T5::kLayers + 1];
    pixelQuintupletsInGPU.logicalLayers[Params_pT5::kLayers * pixelQuintupletIndex + 4] =
        quintupletsInGPU.logicalLayers[T5Index * Params_T5::kLayers + 2];
    pixelQuintupletsInGPU.logicalLayers[Params_pT5::kLayers * pixelQuintupletIndex + 5] =
        quintupletsInGPU.logicalLayers[T5Index * Params_T5::kLayers + 3];
    pixelQuintupletsInGPU.logicalLayers[Params_pT5::kLayers * pixelQuintupletIndex + 6] =
        quintupletsInGPU.logicalLayers[T5Index * Params_T5::kLayers + 4];

    pixelQuintupletsInGPU.lowerModuleIndices[Params_pT5::kLayers * pixelQuintupletIndex] =
        segmentsInGPU.innerLowerModuleIndices[pixelIndex];
    pixelQuintupletsInGPU.lowerModuleIndices[Params_pT5::kLayers * pixelQuintupletIndex + 1] =
        segmentsInGPU.outerLowerModuleIndices[pixelIndex];
    pixelQuintupletsInGPU.lowerModuleIndices[Params_pT5::kLayers * pixelQuintupletIndex + 2] =
        quintupletsInGPU.lowerModuleIndices[T5Index * Params_T5::kLayers];
    pixelQuintupletsInGPU.lowerModuleIndices[Params_pT5::kLayers * pixelQuintupletIndex + 3] =
        quintupletsInGPU.lowerModuleIndices[T5Index * Params_T5::kLayers + 1];
    pixelQuintupletsInGPU.lowerModuleIndices[Params_pT5::kLayers * pixelQuintupletIndex + 4] =
        quintupletsInGPU.lowerModuleIndices[T5Index * Params_T5::kLayers + 2];
    pixelQuintupletsInGPU.lowerModuleIndices[Params_pT5::kLayers * pixelQuintupletIndex + 5] =
        quintupletsInGPU.lowerModuleIndices[T5Index * Params_T5::kLayers + 3];
    pixelQuintupletsInGPU.lowerModuleIndices[Params_pT5::kLayers * pixelQuintupletIndex + 6] =
        quintupletsInGPU.lowerModuleIndices[T5Index * Params_T5::kLayers + 4];

    unsigned int pixelInnerMD = segmentsInGPU.mdIndices[Params_pLS::kLayers * pixelIndex];
    unsigned int pixelOuterMD = segmentsInGPU.mdIndices[Params_pLS::kLayers * pixelIndex + 1];

    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex] =
        mdsInGPU.anchorHitIndices[pixelInnerMD];
    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex + 1] =
        mdsInGPU.outerHitIndices[pixelInnerMD];
    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex + 2] =
        mdsInGPU.anchorHitIndices[pixelOuterMD];
    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex + 3] =
        mdsInGPU.outerHitIndices[pixelOuterMD];

    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex + 4] =
        quintupletsInGPU.hitIndices[Params_T5::kHits * T5Index];
    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex + 5] =
        quintupletsInGPU.hitIndices[Params_T5::kHits * T5Index + 1];
    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex + 6] =
        quintupletsInGPU.hitIndices[Params_T5::kHits * T5Index + 2];
    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex + 7] =
        quintupletsInGPU.hitIndices[Params_T5::kHits * T5Index + 3];
    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex + 8] =
        quintupletsInGPU.hitIndices[Params_T5::kHits * T5Index + 4];
    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex + 9] =
        quintupletsInGPU.hitIndices[Params_T5::kHits * T5Index + 5];
    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex + 10] =
        quintupletsInGPU.hitIndices[Params_T5::kHits * T5Index + 6];
    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex + 11] =
        quintupletsInGPU.hitIndices[Params_T5::kHits * T5Index + 7];
    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex + 12] =
        quintupletsInGPU.hitIndices[Params_T5::kHits * T5Index + 8];
    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex + 13] =
        quintupletsInGPU.hitIndices[Params_T5::kHits * T5Index + 9];

    pixelQuintupletsInGPU.rzChiSquared[pixelQuintupletIndex] = rzChiSquared;
    pixelQuintupletsInGPU.rPhiChiSquared[pixelQuintupletIndex] = rPhiChiSquared;
    pixelQuintupletsInGPU.rPhiChiSquaredInwards[pixelQuintupletIndex] = rPhiChiSquaredInwards;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT5RZChiSquaredCuts(Modules const& modulesInGPU,
                                                              uint16_t lowerModuleIndex1,
                                                              uint16_t lowerModuleIndex2,
                                                              uint16_t lowerModuleIndex3,
                                                              uint16_t lowerModuleIndex4,
                                                              uint16_t lowerModuleIndex5,
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
    const int layer4 =
        modulesInGPU.layers[lowerModuleIndex4] + 6 * (modulesInGPU.subdets[lowerModuleIndex4] == Endcap) +
        5 * (modulesInGPU.subdets[lowerModuleIndex4] == Endcap and modulesInGPU.moduleType[lowerModuleIndex4] == TwoS);
    const int layer5 =
        modulesInGPU.layers[lowerModuleIndex5] + 6 * (modulesInGPU.subdets[lowerModuleIndex5] == Endcap) +
        5 * (modulesInGPU.subdets[lowerModuleIndex5] == Endcap and modulesInGPU.moduleType[lowerModuleIndex5] == TwoS);

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
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT5RPhiChiSquaredCuts(Modules const& modulesInGPU,
                                                                uint16_t lowerModuleIndex1,
                                                                uint16_t lowerModuleIndex2,
                                                                uint16_t lowerModuleIndex3,
                                                                uint16_t lowerModuleIndex4,
                                                                uint16_t lowerModuleIndex5,
                                                                float rPhiChiSquared) {
    const int layer1 =
        modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == Endcap) +
        5 * (modulesInGPU.subdets[lowerModuleIndex1] == Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == TwoS);
    const int layer2 =
        modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == Endcap) +
        5 * (modulesInGPU.subdets[lowerModuleIndex2] == Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == TwoS);
    const int layer3 =
        modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == Endcap) +
        5 * (modulesInGPU.subdets[lowerModuleIndex3] == Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == TwoS);
    const int layer4 =
        modulesInGPU.layers[lowerModuleIndex4] + 6 * (modulesInGPU.subdets[lowerModuleIndex4] == Endcap) +
        5 * (modulesInGPU.subdets[lowerModuleIndex4] == Endcap and modulesInGPU.moduleType[lowerModuleIndex4] == TwoS);
    const int layer5 =
        modulesInGPU.layers[lowerModuleIndex5] + 6 * (modulesInGPU.subdets[lowerModuleIndex5] == Endcap) +
        5 * (modulesInGPU.subdets[lowerModuleIndex5] == Endcap and modulesInGPU.moduleType[lowerModuleIndex5] == TwoS);

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
                    (xs[i] * xs[i] + ys[i] * ys[i] - 2 * g * xs[i] - 2 * f * ys[i] + c) / (sigma2);
    }
    return chiSquared;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void computeSigmasForRegression_pT5(TAcc const& acc,
                                                                     Modules const& modulesInGPU,
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
    float inv1 = kWidthPS / kWidth2S;
    float inv2 = kPixelPSZpitch / kWidth2S;
    float inv3 = kStripPSZpitch / kWidth2S;
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
#ifdef WARNINGS
      else {
        printf("ERROR!!!!! I SHOULDN'T BE HERE!!!! subdet = %d, type = %d, side = %d\n",
               moduleSubdet,
               moduleType,
               moduleSide);
      }
#endif
    }
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computePT5RPhiChiSquared(TAcc const& acc,
                                                                Modules const& modulesInGPU,
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

    computeSigmasForRegression_pT5(acc, modulesInGPU, lowerModuleIndices, delta1, delta2, slopes, isFlat);
    chiSquared = computeChiSquaredpT5(acc, 5, xs, ys, delta1, delta2, slopes, isFlat, g, f, radius);

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

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT5RPhiChiSquaredInwardsCuts(Modules const& modulesInGPU,
                                                                       uint16_t lowerModuleIndex1,
                                                                       uint16_t lowerModuleIndex2,
                                                                       uint16_t lowerModuleIndex3,
                                                                       uint16_t lowerModuleIndex4,
                                                                       uint16_t lowerModuleIndex5,
                                                                       float rPhiChiSquared) {
    const int layer1 =
        modulesInGPU.layers[lowerModuleIndex1] + 6 * (modulesInGPU.subdets[lowerModuleIndex1] == Endcap) +
        5 * (modulesInGPU.subdets[lowerModuleIndex1] == Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == TwoS);
    const int layer2 =
        modulesInGPU.layers[lowerModuleIndex2] + 6 * (modulesInGPU.subdets[lowerModuleIndex2] == Endcap) +
        5 * (modulesInGPU.subdets[lowerModuleIndex2] == Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == TwoS);
    const int layer3 =
        modulesInGPU.layers[lowerModuleIndex3] + 6 * (modulesInGPU.subdets[lowerModuleIndex3] == Endcap) +
        5 * (modulesInGPU.subdets[lowerModuleIndex3] == Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == TwoS);
    const int layer4 =
        modulesInGPU.layers[lowerModuleIndex4] + 6 * (modulesInGPU.subdets[lowerModuleIndex4] == Endcap) +
        5 * (modulesInGPU.subdets[lowerModuleIndex4] == Endcap and modulesInGPU.moduleType[lowerModuleIndex4] == TwoS);
    const int layer5 =
        modulesInGPU.layers[lowerModuleIndex5] + 6 * (modulesInGPU.subdets[lowerModuleIndex5] == Endcap) +
        5 * (modulesInGPU.subdets[lowerModuleIndex5] == Endcap and modulesInGPU.moduleType[lowerModuleIndex5] == TwoS);

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
                                                              Modules const& modulesInGPU,
                                                              uint16_t* lowerModuleIndices,
                                                              float* rtPix,
                                                              float* zPix,
                                                              float* rts,
                                                              float* zs) {
    //use the two anchor hits of the pixel segment to compute the slope
    //then compute the pseudo chi squared of the five outer hits

    float slope = (zPix[1] - zPix[0]) / (rtPix[1] - rtPix[0]);
    float residual = 0;
    float error2 = 0;
    //hardcoded array indices!!!
    float RMSE = 0;
    for (size_t i = 0; i < Params_T5::kLayers; i++) {
      uint16_t& lowerModuleIndex = lowerModuleIndices[i];
      const int moduleType = modulesInGPU.moduleType[lowerModuleIndex];
      const int moduleSide = modulesInGPU.sides[lowerModuleIndex];
      const int moduleSubdet = modulesInGPU.subdets[lowerModuleIndex];

      residual = (moduleSubdet == Barrel) ? (zs[i] - zPix[0]) - slope * (rts[i] - rtPix[0])
                                          : (rts[i] - rtPix[0]) - (zs[i] - zPix[0]) / slope;
      const float& drdz = modulesInGPU.drdzs[lowerModuleIndex];
      //PS Modules
      if (moduleType == 0) {
        error2 = kPixelPSZpitch * kPixelPSZpitch;
      } else  //2S modules
      {
        error2 = kStrip2SZpitch * kStrip2SZpitch;
      }

      //special dispensation to tilted PS modules!
      if (moduleType == 0 and moduleSubdet == Barrel and moduleSide != Center) {
        error2 /= (1.f + drdz * drdz);
      }
      RMSE += (residual * residual) / error2;
    }

    RMSE = alpaka::math::sqrt(acc, 0.2f * RMSE);  // Divided by the degree of freedom 5.
    return RMSE;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runPixelQuintupletDefaultAlgo(TAcc const& acc,
                                                                    Modules const& modulesInGPU,
                                                                    ObjectRanges const& rangesInGPU,
                                                                    MiniDoublets const& mdsInGPU,
                                                                    Segments const& segmentsInGPU,
                                                                    Triplets const& tripletsInGPU,
                                                                    Quintuplets const& quintupletsInGPU,
                                                                    unsigned int pixelSegmentIndex,
                                                                    unsigned int quintupletIndex,
                                                                    float& rzChiSquared,
                                                                    float& rPhiChiSquared,
                                                                    float& rPhiChiSquaredInwards,
                                                                    float& pixelRadius,
                                                                    float& quintupletRadius,
                                                                    float& centerX,
                                                                    float& centerY,
                                                                    unsigned int pixelSegmentArrayIndex) {
    unsigned int T5InnerT3Index = quintupletsInGPU.tripletIndices[2 * quintupletIndex];
    unsigned int T5OuterT3Index = quintupletsInGPU.tripletIndices[2 * quintupletIndex + 1];

    float pixelRadiusTemp, tripletRadius, rPhiChiSquaredTemp, rzChiSquaredTemp, rPhiChiSquaredInwardsTemp, centerXTemp,
        centerYTemp;

    if (not runPixelTripletDefaultAlgo(acc,
                                       modulesInGPU,
                                       rangesInGPU,
                                       mdsInGPU,
                                       segmentsInGPU,
                                       tripletsInGPU,
                                       pixelSegmentIndex,
                                       T5InnerT3Index,
                                       pixelRadiusTemp,
                                       tripletRadius,
                                       centerXTemp,
                                       centerYTemp,
                                       rzChiSquaredTemp,
                                       rPhiChiSquaredTemp,
                                       rPhiChiSquaredInwardsTemp,
                                       false))
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

    uint16_t lowerModuleIndex1 = quintupletsInGPU.lowerModuleIndices[Params_T5::kLayers * quintupletIndex];
    uint16_t lowerModuleIndex2 = quintupletsInGPU.lowerModuleIndices[Params_T5::kLayers * quintupletIndex + 1];
    uint16_t lowerModuleIndex3 = quintupletsInGPU.lowerModuleIndices[Params_T5::kLayers * quintupletIndex + 2];
    uint16_t lowerModuleIndex4 = quintupletsInGPU.lowerModuleIndices[Params_T5::kLayers * quintupletIndex + 3];
    uint16_t lowerModuleIndex5 = quintupletsInGPU.lowerModuleIndices[Params_T5::kLayers * quintupletIndex + 4];

    uint16_t lowerModuleIndices[Params_T5::kLayers] = {
        lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5};

    float zPix[Params_pLS::kLayers] = {mdsInGPU.anchorZ[pixelInnerMDIndex], mdsInGPU.anchorZ[pixelOuterMDIndex]};
    float rtPix[Params_pLS::kLayers] = {mdsInGPU.anchorRt[pixelInnerMDIndex], mdsInGPU.anchorRt[pixelOuterMDIndex]};
    float zs[Params_T5::kLayers] = {mdsInGPU.anchorZ[firstMDIndex],
                                    mdsInGPU.anchorZ[secondMDIndex],
                                    mdsInGPU.anchorZ[thirdMDIndex],
                                    mdsInGPU.anchorZ[fourthMDIndex],
                                    mdsInGPU.anchorZ[fifthMDIndex]};
    float rts[Params_T5::kLayers] = {mdsInGPU.anchorRt[firstMDIndex],
                                     mdsInGPU.anchorRt[secondMDIndex],
                                     mdsInGPU.anchorRt[thirdMDIndex],
                                     mdsInGPU.anchorRt[fourthMDIndex],
                                     mdsInGPU.anchorRt[fifthMDIndex]};

    rzChiSquared = computePT5RZChiSquared(acc, modulesInGPU, lowerModuleIndices, rtPix, zPix, rts, zs);

    if (/*pixelRadius*/ 0 < 5.0f * kR1GeVf) {  // FIXME: pixelRadius is not defined yet
      if (not passPT5RZChiSquaredCuts(modulesInGPU,
                                      lowerModuleIndex1,
                                      lowerModuleIndex2,
                                      lowerModuleIndex3,
                                      lowerModuleIndex4,
                                      lowerModuleIndex5,
                                      rzChiSquared))
        return false;
    }

    //outer T5
    float xs[Params_T5::kLayers] = {mdsInGPU.anchorX[firstMDIndex],
                                    mdsInGPU.anchorX[secondMDIndex],
                                    mdsInGPU.anchorX[thirdMDIndex],
                                    mdsInGPU.anchorX[fourthMDIndex],
                                    mdsInGPU.anchorX[fifthMDIndex]};
    float ys[Params_T5::kLayers] = {mdsInGPU.anchorY[firstMDIndex],
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
      if (not passPT5RPhiChiSquaredCuts(modulesInGPU,
                                        lowerModuleIndex1,
                                        lowerModuleIndex2,
                                        lowerModuleIndex3,
                                        lowerModuleIndex4,
                                        lowerModuleIndex5,
                                        rPhiChiSquared))
        return false;
    }

    float xPix[] = {mdsInGPU.anchorX[pixelInnerMDIndex], mdsInGPU.anchorX[pixelOuterMDIndex]};
    float yPix[] = {mdsInGPU.anchorY[pixelInnerMDIndex], mdsInGPU.anchorY[pixelOuterMDIndex]};
    rPhiChiSquaredInwards = computePT5RPhiChiSquaredInwards(T5CenterX, T5CenterY, quintupletRadius, xPix, yPix);

    if (quintupletsInGPU.regressionRadius[quintupletIndex] < 5.0f * kR1GeVf) {
      if (not passPT5RPhiChiSquaredInwardsCuts(modulesInGPU,
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

  struct CreatePixelQuintupletsInGPUFromMapv2 {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  Modules modulesInGPU,
                                  MiniDoublets mdsInGPU,
                                  Segments segmentsInGPU,
                                  Triplets tripletsInGPU,
                                  Quintuplets quintupletsInGPU,
                                  PixelQuintuplets pixelQuintupletsInGPU,
                                  unsigned int* connectedPixelSize,
                                  unsigned int* connectedPixelIndex,
                                  unsigned int nPixelSegments,
                                  ObjectRanges rangesInGPU) const {
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
          if (modulesInGPU.moduleType[quintupletLowerModuleIndex] == TwoS)
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
              unsigned int totOccupancyPixelQuintuplets = alpaka::atomicAdd(
                  acc, pixelQuintupletsInGPU.totOccupancyPixelQuintuplets, 1u, alpaka::hierarchy::Threads{});
              if (totOccupancyPixelQuintuplets >= n_max_pixel_quintuplets) {
#ifdef WARNINGS
                printf("Pixel Quintuplet excess alert!\n");
#endif
              } else {
                unsigned int pixelQuintupletIndex =
                    alpaka::atomicAdd(acc, pixelQuintupletsInGPU.nPixelQuintuplets, 1u, alpaka::hierarchy::Threads{});
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
            }  // end success
          }  // end T5
        }  // end iLS
      }  // end i_pLS
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
