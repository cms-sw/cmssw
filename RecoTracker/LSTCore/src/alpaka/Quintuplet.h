#ifndef RecoTracker_LSTCore_src_alpaka_Quintuplet_h
#define RecoTracker_LSTCore_src_alpaka_Quintuplet_h

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/Module.h"
#include "RecoTracker/LSTCore/interface/EndcapGeometry.h"

#include "NeuralNetwork.h"
#include "Segment.h"
#include "MiniDoublet.h"
#include "Hit.h"
#include "ObjectRanges.h"
#include "Triplet.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  struct Quintuplets {
    unsigned int* tripletIndices;
    uint16_t* lowerModuleIndices;
    unsigned int* nQuintuplets;
    unsigned int* totOccupancyQuintuplets;
    unsigned int* nMemoryLocations;

    FPX* innerRadius;
    FPX* bridgeRadius;
    FPX* outerRadius;
    FPX* pt;
    FPX* eta;
    FPX* phi;
    FPX* score_rphisum;
    uint8_t* layer;
    char* isDup;
    bool* TightCutFlag;
    bool* partOfPT5;

    float* regressionRadius;
    float* regressionG;
    float* regressionF;

    uint8_t* logicalLayers;
    unsigned int* hitIndices;
    float* rzChiSquared;
    float* chiSquared;
    float* nonAnchorChiSquared;

    template <typename TBuff>
    void setData(TBuff& buf) {
      tripletIndices = buf.tripletIndices_buf.data();
      lowerModuleIndices = buf.lowerModuleIndices_buf.data();
      nQuintuplets = buf.nQuintuplets_buf.data();
      totOccupancyQuintuplets = buf.totOccupancyQuintuplets_buf.data();
      nMemoryLocations = buf.nMemoryLocations_buf.data();
      innerRadius = buf.innerRadius_buf.data();
      bridgeRadius = buf.bridgeRadius_buf.data();
      outerRadius = buf.outerRadius_buf.data();
      pt = buf.pt_buf.data();
      eta = buf.eta_buf.data();
      phi = buf.phi_buf.data();
      score_rphisum = buf.score_rphisum_buf.data();
      layer = buf.layer_buf.data();
      isDup = buf.isDup_buf.data();
      TightCutFlag = buf.TightCutFlag_buf.data();
      partOfPT5 = buf.partOfPT5_buf.data();
      regressionRadius = buf.regressionRadius_buf.data();
      regressionG = buf.regressionG_buf.data();
      regressionF = buf.regressionF_buf.data();
      logicalLayers = buf.logicalLayers_buf.data();
      hitIndices = buf.hitIndices_buf.data();
      rzChiSquared = buf.rzChiSquared_buf.data();
      chiSquared = buf.chiSquared_buf.data();
      nonAnchorChiSquared = buf.nonAnchorChiSquared_buf.data();
    }
  };

  template <typename TDev>
  struct QuintupletsBuffer {
    Buf<TDev, unsigned int> tripletIndices_buf;
    Buf<TDev, uint16_t> lowerModuleIndices_buf;
    Buf<TDev, unsigned int> nQuintuplets_buf;
    Buf<TDev, unsigned int> totOccupancyQuintuplets_buf;
    Buf<TDev, unsigned int> nMemoryLocations_buf;

    Buf<TDev, FPX> innerRadius_buf;
    Buf<TDev, FPX> bridgeRadius_buf;
    Buf<TDev, FPX> outerRadius_buf;
    Buf<TDev, FPX> pt_buf;
    Buf<TDev, FPX> eta_buf;
    Buf<TDev, FPX> phi_buf;
    Buf<TDev, FPX> score_rphisum_buf;
    Buf<TDev, uint8_t> layer_buf;
    Buf<TDev, char> isDup_buf;
    Buf<TDev, bool> TightCutFlag_buf;
    Buf<TDev, bool> partOfPT5_buf;

    Buf<TDev, float> regressionRadius_buf;
    Buf<TDev, float> regressionG_buf;
    Buf<TDev, float> regressionF_buf;

    Buf<TDev, uint8_t> logicalLayers_buf;
    Buf<TDev, unsigned int> hitIndices_buf;
    Buf<TDev, float> rzChiSquared_buf;
    Buf<TDev, float> chiSquared_buf;
    Buf<TDev, float> nonAnchorChiSquared_buf;

    Quintuplets data_;

    template <typename TQueue, typename TDevAcc>
    QuintupletsBuffer(unsigned int nTotalQuintuplets, unsigned int nLowerModules, TDevAcc const& devAccIn, TQueue& queue)
        : tripletIndices_buf(allocBufWrapper<unsigned int>(devAccIn, 2 * nTotalQuintuplets, queue)),
          lowerModuleIndices_buf(allocBufWrapper<uint16_t>(devAccIn, Params_T5::kLayers * nTotalQuintuplets, queue)),
          nQuintuplets_buf(allocBufWrapper<unsigned int>(devAccIn, nLowerModules, queue)),
          totOccupancyQuintuplets_buf(allocBufWrapper<unsigned int>(devAccIn, nLowerModules, queue)),
          nMemoryLocations_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          innerRadius_buf(allocBufWrapper<FPX>(devAccIn, nTotalQuintuplets, queue)),
          bridgeRadius_buf(allocBufWrapper<FPX>(devAccIn, nTotalQuintuplets, queue)),
          outerRadius_buf(allocBufWrapper<FPX>(devAccIn, nTotalQuintuplets, queue)),
          pt_buf(allocBufWrapper<FPX>(devAccIn, nTotalQuintuplets, queue)),
          eta_buf(allocBufWrapper<FPX>(devAccIn, nTotalQuintuplets, queue)),
          phi_buf(allocBufWrapper<FPX>(devAccIn, nTotalQuintuplets, queue)),
          score_rphisum_buf(allocBufWrapper<FPX>(devAccIn, nTotalQuintuplets, queue)),
          layer_buf(allocBufWrapper<uint8_t>(devAccIn, nTotalQuintuplets, queue)),
          isDup_buf(allocBufWrapper<char>(devAccIn, nTotalQuintuplets, queue)),
          TightCutFlag_buf(allocBufWrapper<bool>(devAccIn, nTotalQuintuplets, queue)),
          partOfPT5_buf(allocBufWrapper<bool>(devAccIn, nTotalQuintuplets, queue)),
          regressionRadius_buf(allocBufWrapper<float>(devAccIn, nTotalQuintuplets, queue)),
          regressionG_buf(allocBufWrapper<float>(devAccIn, nTotalQuintuplets, queue)),
          regressionF_buf(allocBufWrapper<float>(devAccIn, nTotalQuintuplets, queue)),
          logicalLayers_buf(allocBufWrapper<uint8_t>(devAccIn, Params_T5::kLayers * nTotalQuintuplets, queue)),
          hitIndices_buf(allocBufWrapper<unsigned int>(devAccIn, Params_T5::kHits * nTotalQuintuplets, queue)),
          rzChiSquared_buf(allocBufWrapper<float>(devAccIn, nTotalQuintuplets, queue)),
          chiSquared_buf(allocBufWrapper<float>(devAccIn, nTotalQuintuplets, queue)),
          nonAnchorChiSquared_buf(allocBufWrapper<float>(devAccIn, nTotalQuintuplets, queue)) {
      alpaka::memset(queue, nQuintuplets_buf, 0u);
      alpaka::memset(queue, totOccupancyQuintuplets_buf, 0u);
      alpaka::memset(queue, isDup_buf, 0u);
      alpaka::memset(queue, TightCutFlag_buf, false);
      alpaka::memset(queue, partOfPT5_buf, false);
    }

    inline Quintuplets const* data() const { return &data_; }
    inline void setData(QuintupletsBuffer& buf) { data_.setData(buf); }
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool checkIntervalOverlap(float firstMin,
                                                           float firstMax,
                                                           float secondMin,
                                                           float secondMax) {
    return ((firstMin <= secondMin) && (secondMin < firstMax)) || ((secondMin < firstMin) && (firstMin < secondMax));
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addQuintupletToMemory(Triplets const& tripletsInGPU,
                                                            Quintuplets& quintupletsInGPU,
                                                            unsigned int innerTripletIndex,
                                                            unsigned int outerTripletIndex,
                                                            uint16_t lowerModule1,
                                                            uint16_t lowerModule2,
                                                            uint16_t lowerModule3,
                                                            uint16_t lowerModule4,
                                                            uint16_t lowerModule5,
                                                            float innerRadius,
                                                            float bridgeRadius,
                                                            float outerRadius,
                                                            float regressionG,
                                                            float regressionF,
                                                            float regressionRadius,
                                                            float rzChiSquared,
                                                            float rPhiChiSquared,
                                                            float nonAnchorChiSquared,
                                                            float pt,
                                                            float eta,
                                                            float phi,
                                                            float scores,
                                                            uint8_t layer,
                                                            unsigned int quintupletIndex,
                                                            bool TightCutFlag) {
    quintupletsInGPU.tripletIndices[2 * quintupletIndex] = innerTripletIndex;
    quintupletsInGPU.tripletIndices[2 * quintupletIndex + 1] = outerTripletIndex;

    quintupletsInGPU.lowerModuleIndices[Params_T5::kLayers * quintupletIndex] = lowerModule1;
    quintupletsInGPU.lowerModuleIndices[Params_T5::kLayers * quintupletIndex + 1] = lowerModule2;
    quintupletsInGPU.lowerModuleIndices[Params_T5::kLayers * quintupletIndex + 2] = lowerModule3;
    quintupletsInGPU.lowerModuleIndices[Params_T5::kLayers * quintupletIndex + 3] = lowerModule4;
    quintupletsInGPU.lowerModuleIndices[Params_T5::kLayers * quintupletIndex + 4] = lowerModule5;
    quintupletsInGPU.innerRadius[quintupletIndex] = __F2H(innerRadius);
    quintupletsInGPU.outerRadius[quintupletIndex] = __F2H(outerRadius);
    quintupletsInGPU.pt[quintupletIndex] = __F2H(pt);
    quintupletsInGPU.eta[quintupletIndex] = __F2H(eta);
    quintupletsInGPU.phi[quintupletIndex] = __F2H(phi);
    quintupletsInGPU.score_rphisum[quintupletIndex] = __F2H(scores);
    quintupletsInGPU.layer[quintupletIndex] = layer;
    quintupletsInGPU.isDup[quintupletIndex] = 0;
    quintupletsInGPU.TightCutFlag[quintupletIndex] = TightCutFlag;
    quintupletsInGPU.regressionRadius[quintupletIndex] = regressionRadius;
    quintupletsInGPU.regressionG[quintupletIndex] = regressionG;
    quintupletsInGPU.regressionF[quintupletIndex] = regressionF;
    quintupletsInGPU.logicalLayers[Params_T5::kLayers * quintupletIndex] =
        tripletsInGPU.logicalLayers[Params_T3::kLayers * innerTripletIndex];
    quintupletsInGPU.logicalLayers[Params_T5::kLayers * quintupletIndex + 1] =
        tripletsInGPU.logicalLayers[Params_T3::kLayers * innerTripletIndex + 1];
    quintupletsInGPU.logicalLayers[Params_T5::kLayers * quintupletIndex + 2] =
        tripletsInGPU.logicalLayers[Params_T3::kLayers * innerTripletIndex + 2];
    quintupletsInGPU.logicalLayers[Params_T5::kLayers * quintupletIndex + 3] =
        tripletsInGPU.logicalLayers[Params_T3::kLayers * outerTripletIndex + 1];
    quintupletsInGPU.logicalLayers[Params_T5::kLayers * quintupletIndex + 4] =
        tripletsInGPU.logicalLayers[Params_T3::kLayers * outerTripletIndex + 2];

    quintupletsInGPU.hitIndices[Params_T5::kHits * quintupletIndex] =
        tripletsInGPU.hitIndices[Params_T3::kHits * innerTripletIndex];
    quintupletsInGPU.hitIndices[Params_T5::kHits * quintupletIndex + 1] =
        tripletsInGPU.hitIndices[Params_T3::kHits * innerTripletIndex + 1];
    quintupletsInGPU.hitIndices[Params_T5::kHits * quintupletIndex + 2] =
        tripletsInGPU.hitIndices[Params_T3::kHits * innerTripletIndex + 2];
    quintupletsInGPU.hitIndices[Params_T5::kHits * quintupletIndex + 3] =
        tripletsInGPU.hitIndices[Params_T3::kHits * innerTripletIndex + 3];
    quintupletsInGPU.hitIndices[Params_T5::kHits * quintupletIndex + 4] =
        tripletsInGPU.hitIndices[Params_T3::kHits * innerTripletIndex + 4];
    quintupletsInGPU.hitIndices[Params_T5::kHits * quintupletIndex + 5] =
        tripletsInGPU.hitIndices[Params_T3::kHits * innerTripletIndex + 5];
    quintupletsInGPU.hitIndices[Params_T5::kHits * quintupletIndex + 6] =
        tripletsInGPU.hitIndices[Params_T3::kHits * outerTripletIndex + 2];
    quintupletsInGPU.hitIndices[Params_T5::kHits * quintupletIndex + 7] =
        tripletsInGPU.hitIndices[Params_T3::kHits * outerTripletIndex + 3];
    quintupletsInGPU.hitIndices[Params_T5::kHits * quintupletIndex + 8] =
        tripletsInGPU.hitIndices[Params_T3::kHits * outerTripletIndex + 4];
    quintupletsInGPU.hitIndices[Params_T5::kHits * quintupletIndex + 9] =
        tripletsInGPU.hitIndices[Params_T3::kHits * outerTripletIndex + 5];
    quintupletsInGPU.bridgeRadius[quintupletIndex] = bridgeRadius;
    quintupletsInGPU.rzChiSquared[quintupletIndex] = rzChiSquared;
    quintupletsInGPU.chiSquared[quintupletIndex] = rPhiChiSquared;
    quintupletsInGPU.nonAnchorChiSquared[quintupletIndex] = nonAnchorChiSquared;
  }

  //90% constraint
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passChiSquaredConstraint(Modules const& modulesInGPU,
                                                               uint16_t lowerModuleIndex1,
                                                               uint16_t lowerModuleIndex2,
                                                               uint16_t lowerModuleIndex3,
                                                               uint16_t lowerModuleIndex4,
                                                               uint16_t lowerModuleIndex5,
                                                               float chiSquared) {
    // Using lstLayer numbering convention defined in ModuleMethods.h
    const int layer1 = modulesInGPU.lstLayers[lowerModuleIndex1];
    const int layer2 = modulesInGPU.lstLayers[lowerModuleIndex2];
    const int layer3 = modulesInGPU.lstLayers[lowerModuleIndex3];
    const int layer4 = modulesInGPU.lstLayers[lowerModuleIndex4];
    const int layer5 = modulesInGPU.lstLayers[lowerModuleIndex5];

    if (layer1 == 7 and layer2 == 8 and layer3 == 9) {
      if (layer4 == 10 and layer5 == 11) {
        return chiSquared < 0.01788f;
      } else if (layer4 == 10 and layer5 == 16) {
        return chiSquared < 0.04725f;
      } else if (layer4 == 15 and layer5 == 16) {
        return chiSquared < 0.04725f;
      }
    } else if (layer1 == 1 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 9 and layer5 == 10) {
        return chiSquared < 0.01788f;
      } else if (layer4 == 9 and layer5 == 15) {
        return chiSquared < 0.08234f;
      }
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 7) {
      if (layer4 == 8 and layer5 == 9) {
        return chiSquared < 0.02360f;
      } else if (layer4 == 8 and layer5 == 14) {
        return chiSquared < 0.07167f;
      } else if (layer4 == 13 and layer5 == 14) {
        return chiSquared < 0.08234f;
      }
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 3) {
      if (layer4 == 7 and layer5 == 8) {
        return chiSquared < 0.01026f;
      } else if (layer4 == 7 and layer5 == 13) {
        return chiSquared < 0.06238f;
      } else if (layer4 == 12 and layer5 == 13) {
        return chiSquared < 0.06238f;
      }
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 3 and layer4 == 4) {
      if (layer5 == 5) {
        return chiSquared < 0.04725f;
      } else if (layer5 == 12) {
        return chiSquared < 0.09461f;
      }
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 9 and layer5 == 10) {
        return chiSquared < 0.00512f;
      }
      if (layer4 == 9 and layer5 == 15) {
        return chiSquared < 0.04112f;
      } else if (layer4 == 14 and layer5 == 15) {
        return chiSquared < 0.06238f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 7) {
      if (layer4 == 8 and layer5 == 14) {
        return chiSquared < 0.07167f;
      } else if (layer4 == 13 and layer5 == 14) {
        return chiSquared < 0.06238f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 4) {
      if (layer4 == 5 and layer5 == 6) {
        return chiSquared < 0.08234f;
      } else if (layer4 == 5 and layer5 == 12) {
        return chiSquared < 0.10870f;
      } else if (layer4 == 12 and layer5 == 13) {
        return chiSquared < 0.10870f;
      }
    } else if (layer1 == 3 and layer2 == 7 and layer3 == 8 and layer4 == 14 and layer5 == 15) {
      return chiSquared < 0.09461f;
    } else if (layer1 == 3 and layer2 == 4 and layer3 == 5 and layer4 == 12 and layer5 == 13) {
      return chiSquared < 0.09461f;
    }

    return true;
  }

  //bounds can be found at http://uaf-10.t2.ucsd.edu/~bsathian/SDL/T5_RZFix/t5_rz_thresholds.txt
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passT5RZConstraint(TAcc const& acc,
                                                         Modules const& modulesInGPU,
                                                         MiniDoublets const& mdsInGPU,
                                                         unsigned int firstMDIndex,
                                                         unsigned int secondMDIndex,
                                                         unsigned int thirdMDIndex,
                                                         unsigned int fourthMDIndex,
                                                         unsigned int fifthMDIndex,
                                                         uint16_t lowerModuleIndex1,
                                                         uint16_t lowerModuleIndex2,
                                                         uint16_t lowerModuleIndex3,
                                                         uint16_t lowerModuleIndex4,
                                                         uint16_t lowerModuleIndex5,
                                                         float& rzChiSquared,
                                                         float inner_pt,
                                                         float innerRadius,
                                                         float g,
                                                         float f,
                                                         bool& TightCutFlag) {
    //(g,f) is the center of the circle fitted by the innermost 3 points on x,y coordinates
    const float& rt1 = mdsInGPU.anchorRt[firstMDIndex] / 100;  //in the unit of m instead of cm
    const float& rt2 = mdsInGPU.anchorRt[secondMDIndex] / 100;
    const float& rt3 = mdsInGPU.anchorRt[thirdMDIndex] / 100;
    const float& rt4 = mdsInGPU.anchorRt[fourthMDIndex] / 100;
    const float& rt5 = mdsInGPU.anchorRt[fifthMDIndex] / 100;

    const float& z1 = mdsInGPU.anchorZ[firstMDIndex] / 100;
    const float& z2 = mdsInGPU.anchorZ[secondMDIndex] / 100;
    const float& z3 = mdsInGPU.anchorZ[thirdMDIndex] / 100;
    const float& z4 = mdsInGPU.anchorZ[fourthMDIndex] / 100;
    const float& z5 = mdsInGPU.anchorZ[fifthMDIndex] / 100;

    // Using lst_layer numbering convention defined in ModuleMethods.h
    const int layer1 = modulesInGPU.lstLayers[lowerModuleIndex1];
    const int layer2 = modulesInGPU.lstLayers[lowerModuleIndex2];
    const int layer3 = modulesInGPU.lstLayers[lowerModuleIndex3];
    const int layer4 = modulesInGPU.lstLayers[lowerModuleIndex4];
    const int layer5 = modulesInGPU.lstLayers[lowerModuleIndex5];

    //slope computed using the internal T3s
    const int moduleType1 = modulesInGPU.moduleType[lowerModuleIndex1];  //0 is ps, 1 is 2s
    const int moduleType2 = modulesInGPU.moduleType[lowerModuleIndex2];
    const int moduleType3 = modulesInGPU.moduleType[lowerModuleIndex3];
    const int moduleType4 = modulesInGPU.moduleType[lowerModuleIndex4];
    const int moduleType5 = modulesInGPU.moduleType[lowerModuleIndex5];

    const float& x1 = mdsInGPU.anchorX[firstMDIndex] / 100;
    const float& x2 = mdsInGPU.anchorX[secondMDIndex] / 100;
    const float& x3 = mdsInGPU.anchorX[thirdMDIndex] / 100;
    const float& x4 = mdsInGPU.anchorX[fourthMDIndex] / 100;
    const float& y1 = mdsInGPU.anchorY[firstMDIndex] / 100;
    const float& y2 = mdsInGPU.anchorY[secondMDIndex] / 100;
    const float& y3 = mdsInGPU.anchorY[thirdMDIndex] / 100;
    const float& y4 = mdsInGPU.anchorY[fourthMDIndex] / 100;

    float residual = 0;
    float error2 = 0;
    float x_center = g / 100, y_center = f / 100;
    float x_init = mdsInGPU.anchorX[thirdMDIndex] / 100;
    float y_init = mdsInGPU.anchorY[thirdMDIndex] / 100;
    float z_init = mdsInGPU.anchorZ[thirdMDIndex] / 100;
    float rt_init = mdsInGPU.anchorRt[thirdMDIndex] / 100;  //use the second MD as initial point

    if (moduleType3 == 1)  // 1: if MD3 is in 2s layer
    {
      x_init = mdsInGPU.anchorX[secondMDIndex] / 100;
      y_init = mdsInGPU.anchorY[secondMDIndex] / 100;
      z_init = mdsInGPU.anchorZ[secondMDIndex] / 100;
      rt_init = mdsInGPU.anchorRt[secondMDIndex] / 100;
    }

    // start from a circle of inner T3.
    // to determine the charge
    int charge = 0;
    float slope3c = (y3 - y_center) / (x3 - x_center);
    float slope1c = (y1 - y_center) / (x1 - x_center);
    // these 4 "if"s basically separate the x-y plane into 4 quarters. It determines geometrically how a circle and line slope goes and their positions, and we can get the charges correspondingly.
    if ((y3 - y_center) > 0 && (y1 - y_center) > 0) {
      if (slope1c > 0 && slope3c < 0)
        charge = -1;  // on x axis of a quarter, 3 hits go anti-clockwise
      else if (slope1c < 0 && slope3c > 0)
        charge = 1;  // on x axis of a quarter, 3 hits go clockwise
      else if (slope3c > slope1c)
        charge = -1;
      else if (slope3c < slope1c)
        charge = 1;
    } else if ((y3 - y_center) < 0 && (y1 - y_center) < 0) {
      if (slope1c < 0 && slope3c > 0)
        charge = 1;
      else if (slope1c > 0 && slope3c < 0)
        charge = -1;
      else if (slope3c > slope1c)
        charge = -1;
      else if (slope3c < slope1c)
        charge = 1;
    } else if ((y3 - y_center) < 0 && (y1 - y_center) > 0) {
      if ((x3 - x_center) > 0 && (x1 - x_center) > 0)
        charge = 1;
      else if ((x3 - x_center) < 0 && (x1 - x_center) < 0)
        charge = -1;
    } else if ((y3 - y_center) > 0 && (y1 - y_center) < 0) {
      if ((x3 - x_center) > 0 && (x1 - x_center) > 0)
        charge = -1;
      else if ((x3 - x_center) < 0 && (x1 - x_center) < 0)
        charge = 1;
    }

    float pseudo_phi = alpaka::math::atan(
        acc, (y_init - y_center) / (x_init - x_center));  //actually represent pi/2-phi, wrt helix axis z
    float Pt = inner_pt, Px = Pt * alpaka::math::abs(acc, alpaka::math::sin(acc, pseudo_phi)),
          Py = Pt * alpaka::math::abs(acc, cos(pseudo_phi));

    // Above line only gives you the correct value of Px and Py, but signs of Px and Py calculated below.
    // We look at if the circle is clockwise or anti-clock wise, to make it simpler, we separate the x-y plane into 4 quarters.
    if (x_init > x_center && y_init > y_center)  //1st quad
    {
      if (charge == 1)
        Py = -Py;
      if (charge == -1)
        Px = -Px;
    }
    if (x_init < x_center && y_init > y_center)  //2nd quad
    {
      if (charge == -1) {
        Px = -Px;
        Py = -Py;
      }
    }
    if (x_init < x_center && y_init < y_center)  //3rd quad
    {
      if (charge == 1)
        Px = -Px;
      if (charge == -1)
        Py = -Py;
    }
    if (x_init > x_center && y_init < y_center)  //4th quad
    {
      if (charge == 1) {
        Px = -Px;
        Py = -Py;
      }
    }

    // But if the initial T5 curve goes across quarters(i.e. cross axis to separate the quarters), need special redeclaration of Px,Py signs on these to avoid errors
    if (moduleType3 == 0) {  // 0 is ps
      if (x4 < x3 && x3 < x2)
        Px = -alpaka::math::abs(acc, Px);
      else if (x4 > x3 && x3 > x2)
        Px = alpaka::math::abs(acc, Px);
      if (y4 < y3 && y3 < y2)
        Py = -alpaka::math::abs(acc, Py);
      else if (y4 > y3 && y3 > y2)
        Py = alpaka::math::abs(acc, Py);
    } else if (moduleType3 == 1)  // 1 is 2s
    {
      if (x3 < x2 && x2 < x1)
        Px = -alpaka::math::abs(acc, Px);
      else if (x3 > x2 && x2 > x1)
        Px = alpaka::math::abs(acc, Px);
      if (y3 < y2 && y2 < y1)
        Py = -alpaka::math::abs(acc, Py);
      else if (y3 > y2 && y2 > y1)
        Py = alpaka::math::abs(acc, Py);
    }

    //to get Pz, we use pt/pz=ds/dz, ds is the arclength between MD1 and MD3.
    float AO = alpaka::math::sqrt(acc, (x1 - x_center) * (x1 - x_center) + (y1 - y_center) * (y1 - y_center));
    float BO =
        alpaka::math::sqrt(acc, (x_init - x_center) * (x_init - x_center) + (y_init - y_center) * (y_init - y_center));
    float AB2 = (x1 - x_init) * (x1 - x_init) + (y1 - y_init) * (y1 - y_init);
    float dPhi = alpaka::math::acos(acc, (AO * AO + BO * BO - AB2) / (2 * AO * BO));
    float ds = innerRadius / 100 * dPhi;

    float Pz = (z_init - z1) / ds * Pt;
    float p = alpaka::math::sqrt(acc, Px * Px + Py * Py + Pz * Pz);

    float a = -2.f * k2Rinv1GeVf * 100 * charge;  // multiply by 100 to make the correct length units

    float zsi, rtsi;
    int layeri, moduleTypei;
    rzChiSquared = 0;
    for (size_t i = 2; i < 6; i++) {
      if (i == 2) {
        zsi = z2;
        rtsi = rt2;
        layeri = layer2;
        moduleTypei = moduleType2;
      } else if (i == 3) {
        zsi = z3;
        rtsi = rt3;
        layeri = layer3;
        moduleTypei = moduleType3;
      } else if (i == 4) {
        zsi = z4;
        rtsi = rt4;
        layeri = layer4;
        moduleTypei = moduleType4;
      } else if (i == 5) {
        zsi = z5;
        rtsi = rt5;
        layeri = layer5;
        moduleTypei = moduleType5;
      }

      if (moduleType3 == 0) {  //0: ps
        if (i == 3)
          continue;
      } else {
        if (i == 2)
          continue;
      }

      // calculation is copied from PixelTriplet.cc computePT3RZChiSquared
      float diffr = 0, diffz = 0;

      float rou = a / p;
      // for endcap
      float s = (zsi - z_init) * p / Pz;
      float x = x_init + Px / a * alpaka::math::sin(acc, rou * s) - Py / a * (1 - alpaka::math::cos(acc, rou * s));
      float y = y_init + Py / a * alpaka::math::sin(acc, rou * s) + Px / a * (1 - alpaka::math::cos(acc, rou * s));
      diffr = (rtsi - alpaka::math::sqrt(acc, x * x + y * y)) * 100;

      // for barrel
      if (layeri <= 6) {
        float paraA =
            rt_init * rt_init + 2 * (Px * Px + Py * Py) / (a * a) + 2 * (y_init * Px - x_init * Py) / a - rtsi * rtsi;
        float paraB = 2 * (x_init * Px + y_init * Py) / a;
        float paraC = 2 * (y_init * Px - x_init * Py) / a + 2 * (Px * Px + Py * Py) / (a * a);
        float A = paraB * paraB + paraC * paraC;
        float B = 2 * paraA * paraB;
        float C = paraA * paraA - paraC * paraC;
        float sol1 = (-B + alpaka::math::sqrt(acc, B * B - 4 * A * C)) / (2 * A);
        float sol2 = (-B - alpaka::math::sqrt(acc, B * B - 4 * A * C)) / (2 * A);
        float solz1 = alpaka::math::asin(acc, sol1) / rou * Pz / p + z_init;
        float solz2 = alpaka::math::asin(acc, sol2) / rou * Pz / p + z_init;
        float diffz1 = (solz1 - zsi) * 100;
        float diffz2 = (solz2 - zsi) * 100;
        // Alpaka : Needs to be moved over
        if (alpaka::math::isnan(acc, diffz1))
          diffz = diffz2;
        else if (alpaka::math::isnan(acc, diffz2))
          diffz = diffz1;
        else {
          diffz = (alpaka::math::abs(acc, diffz1) < alpaka::math::abs(acc, diffz2)) ? diffz1 : diffz2;
        }
      }
      residual = (layeri > 6) ? diffr : diffz;

      //PS Modules
      if (moduleTypei == 0) {
        error2 = kPixelPSZpitch * kPixelPSZpitch;
      } else  //2S modules
      {
        error2 = kStrip2SZpitch * kStrip2SZpitch;
      }

      //check the tilted module, side: PosZ, NegZ, Center(for not tilted)
      float drdz;
      short side, subdets;
      if (i == 2) {
        drdz = alpaka::math::abs(acc, modulesInGPU.drdzs[lowerModuleIndex2]);
        side = modulesInGPU.sides[lowerModuleIndex2];
        subdets = modulesInGPU.subdets[lowerModuleIndex2];
      }
      if (i == 3) {
        drdz = alpaka::math::abs(acc, modulesInGPU.drdzs[lowerModuleIndex3]);
        side = modulesInGPU.sides[lowerModuleIndex3];
        subdets = modulesInGPU.subdets[lowerModuleIndex3];
      }
      if (i == 2 || i == 3) {
        residual = (layeri <= 6 && ((side == Center) or (drdz < 1))) ? diffz : diffr;
        float projection_missing2 = 1.f;
        if (drdz < 1)
          projection_missing2 =
              ((subdets == Endcap) or (side == Center)) ? 1.f : 1.f / (1 + drdz * drdz);  // cos(atan(drdz)), if dr/dz<1
        if (drdz > 1)
          projection_missing2 = ((subdets == Endcap) or (side == Center))
                                    ? 1.f
                                    : (drdz * drdz) / (1 + drdz * drdz);  //sin(atan(drdz)), if dr/dz>1
        error2 = error2 * projection_missing2;
      }
      rzChiSquared += 12 * (residual * residual) / error2;
    }
    // for set rzchi2 cut
    // if the 5 points are linear, helix calculation gives nan
    // Alpaka : Needs to be moved over
    if (inner_pt > 100 || alpaka::math::isnan(acc, rzChiSquared)) {
      float slope;
      if (moduleType1 == 0 and moduleType2 == 0 and moduleType3 == 1)  //PSPS2S
      {
        slope = (z2 - z1) / (rt2 - rt1);
      } else {
        slope = (z3 - z1) / (rt3 - rt1);
      }
      float residual4_linear = (layer4 <= 6) ? ((z4 - z1) - slope * (rt4 - rt1)) : ((rt4 - rt1) - (z4 - z1) / slope);
      float residual5_linear = (layer4 <= 6) ? ((z5 - z1) - slope * (rt5 - rt1)) : ((rt5 - rt1) - (z5 - z1) / slope);

      // creating a chi squared type quantity
      // 0-> PS, 1->2S
      residual4_linear = (moduleType4 == 0) ? residual4_linear / kPixelPSZpitch : residual4_linear / kStrip2SZpitch;
      residual5_linear = (moduleType5 == 0) ? residual5_linear / kPixelPSZpitch : residual5_linear / kStrip2SZpitch;
      residual4_linear = residual4_linear * 100;
      residual5_linear = residual5_linear * 100;

      rzChiSquared = 12 * (residual4_linear * residual4_linear + residual5_linear * residual5_linear);
      return rzChiSquared < 4.677f;
    }

    // when building T5, apply 99% chi2 cuts as default, and add to pT5 collection. But when adding T5 to TC collections, apply 95% cut to reduce the fake rate
    TightCutFlag = false;
    // The category numbers are related to module regions and layers, decoding of the region numbers can be found here in slide 2 table. https://github.com/SegmentLinking/TrackLooper/files/11420927/part.2.pdf
    // The commented numbers after each case is the region code, and can look it up from the table to see which category it belongs to. For example, //0 means T5 built with Endcap 1,2,3,4,5 ps modules
    if (layer1 == 7 and layer2 == 8 and layer3 == 9 and layer4 == 10 and layer5 == 11)  //0
    {
      if (rzChiSquared < 94.470f)
        TightCutFlag = true;
      return true;
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 9 and layer4 == 10 and layer5 == 16)  //1
    {
      if (rzChiSquared < 22.099f)
        TightCutFlag = true;
      return rzChiSquared < 37.956f;
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 9 and layer4 == 15 and layer5 == 16)  //2
    {
      if (rzChiSquared < 7.992f)
        TightCutFlag = true;
      return rzChiSquared < 11.622f;
    } else if (layer1 == 1 and layer2 == 7 and layer3 == 8 and layer4 == 9) {
      if (layer5 == 10)  //3
      {
        if (rzChiSquared < 111.390f)
          TightCutFlag = true;
        return true;
      }
      if (layer5 == 15)  //4
      {
        if (rzChiSquared < 18.351f)
          TightCutFlag = true;
        return rzChiSquared < 37.941f;
      }
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 7) {
      if (layer4 == 8 and layer5 == 9)  //5
      {
        if (rzChiSquared < 116.148f)
          TightCutFlag = true;
        return true;
      }
      if (layer4 == 8 and layer5 == 14)  //6
      {
        if (rzChiSquared < 19.352f)
          TightCutFlag = true;
        return rzChiSquared < 52.561f;
      } else if (layer4 == 13 and layer5 == 14)  //7
      {
        if (rzChiSquared < 10.392f)
          TightCutFlag = true;
        return rzChiSquared < 13.76f;
      }
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 3) {
      if (layer4 == 7 and layer5 == 8)  //8
      {
        if (rzChiSquared < 27.824f)
          TightCutFlag = true;
        return rzChiSquared < 44.247f;
      } else if (layer4 == 7 and layer5 == 13)  //9
      {
        if (rzChiSquared < 18.145f)
          TightCutFlag = true;
        return rzChiSquared < 33.752f;
      } else if (layer4 == 12 and layer5 == 13)  //10
      {
        if (rzChiSquared < 13.308f)
          TightCutFlag = true;
        return rzChiSquared < 21.213f;
      } else if (layer4 == 4 and layer5 == 5)  //11
      {
        if (rzChiSquared < 15.627f)
          TightCutFlag = true;
        return rzChiSquared < 29.035f;
      } else if (layer4 == 4 and layer5 == 12)  //12
      {
        if (rzChiSquared < 14.64f)
          TightCutFlag = true;
        return rzChiSquared < 23.037f;
      }
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 9 and layer5 == 15)  //14
      {
        if (rzChiSquared < 24.662f)
          TightCutFlag = true;
        return rzChiSquared < 41.036f;
      } else if (layer4 == 14 and layer5 == 15)  //15
      {
        if (rzChiSquared < 8.866f)
          TightCutFlag = true;
        return rzChiSquared < 14.092f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 7) {
      if (layer4 == 8 and layer5 == 14)  //16
      {
        if (rzChiSquared < 23.730f)
          TightCutFlag = true;
        return rzChiSquared < 23.748f;
      }
      if (layer4 == 13 and layer5 == 14)  //17
      {
        if (rzChiSquared < 10.772f)
          TightCutFlag = true;
        return rzChiSquared < 17.945f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 4) {
      if (layer4 == 5 and layer5 == 6)  //18
      {
        if (rzChiSquared < 6.065f)
          TightCutFlag = true;
        return rzChiSquared < 8.803f;
      } else if (layer4 == 5 and layer5 == 12)  //19
      {
        if (rzChiSquared < 5.693f)
          TightCutFlag = true;
        return rzChiSquared < 7.930f;
      }

      else if (layer4 == 12 and layer5 == 13)  //20
      {
        if (rzChiSquared < 5.473f)
          TightCutFlag = true;
        return rzChiSquared < 7.626f;
      }
    }
    return true;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool T5HasCommonMiniDoublet(Triplets const& tripletsInGPU,
                                                             Segments const& segmentsInGPU,
                                                             unsigned int innerTripletIndex,
                                                             unsigned int outerTripletIndex) {
    unsigned int innerOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex + 1];
    unsigned int outerInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex];
    unsigned int innerOuterOuterMiniDoubletIndex =
        segmentsInGPU.mdIndices[2 * innerOuterSegmentIndex + 1];  //inner triplet outer segment outer MD index
    unsigned int outerInnerInnerMiniDoubletIndex =
        segmentsInGPU.mdIndices[2 * outerInnerSegmentIndex];  //outer triplet inner segment inner MD index

    return (innerOuterOuterMiniDoubletIndex == outerInnerInnerMiniDoubletIndex);
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void computeErrorInRadius(TAcc const& acc,
                                                           float* x1Vec,
                                                           float* y1Vec,
                                                           float* x2Vec,
                                                           float* y2Vec,
                                                           float* x3Vec,
                                                           float* y3Vec,
                                                           float& minimumRadius,
                                                           float& maximumRadius) {
    //brute force
    float candidateRadius;
    float g, f;
    minimumRadius = lst_INF;
    maximumRadius = 0.f;
    for (size_t i = 0; i < 3; i++) {
      float x1 = x1Vec[i];
      float y1 = y1Vec[i];
      for (size_t j = 0; j < 3; j++) {
        float x2 = x2Vec[j];
        float y2 = y2Vec[j];
        for (size_t k = 0; k < 3; k++) {
          float x3 = x3Vec[k];
          float y3 = y3Vec[k];
          candidateRadius = computeRadiusFromThreeAnchorHits(acc, x1, y1, x2, y2, x3, y3, g, f);
          maximumRadius = alpaka::math::max(acc, candidateRadius, maximumRadius);
          minimumRadius = alpaka::math::min(acc, candidateRadius, minimumRadius);
        }
      }
    }
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool matchRadiiBBBEE12378(TAcc const& acc,
                                                           float innerRadius,
                                                           float bridgeRadius,
                                                           float outerRadius,
                                                           float bridgeRadiusMin2S,
                                                           float bridgeRadiusMax2S) {
    float innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax;

    float innerInvRadiusErrorBound = 0.178f;
    float bridgeInvRadiusErrorBound = 0.507f;

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
    bridgeInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

    return checkIntervalOverlap(innerInvRadiusMin,
                                innerInvRadiusMax,
                                alpaka::math::min(acc, bridgeInvRadiusMin, 1.0f / bridgeRadiusMax2S),
                                alpaka::math::max(acc, bridgeInvRadiusMax, 1.0f / bridgeRadiusMin2S));
  }

  /*bounds for high Pt taken from : http://uaf-10.t2.ucsd.edu/~bsathian/SDL/T5_efficiency/efficiencies/new_efficiencies/efficiencies_20210513_T5_recovering_high_Pt_efficiencies/highE_radius_matching/highE_bounds.txt */
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool matchRadiiBBBBB(TAcc const& acc,
                                                      float innerRadius,
                                                      float bridgeRadius,
                                                      float outerRadius) {
    float innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax;

    float innerInvRadiusErrorBound = 0.1512f;
    float bridgeInvRadiusErrorBound = 0.1781f;

    if (innerRadius > 2.0f / (2.f * k2Rinv1GeVf)) {
      innerInvRadiusErrorBound = 0.4449f;
      bridgeInvRadiusErrorBound = 0.4033f;
    }

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
    bridgeInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

    return checkIntervalOverlap(innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax);
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool matchRadiiBBBBE(TAcc const& acc,
                                                      float innerRadius,
                                                      float bridgeRadius,
                                                      float outerRadius) {
    float innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax;

    float innerInvRadiusErrorBound = 0.1781f;
    float bridgeInvRadiusErrorBound = 0.2167f;

    if (innerRadius > 2.0f / (2.f * k2Rinv1GeVf)) {
      innerInvRadiusErrorBound = 0.4750f;
      bridgeInvRadiusErrorBound = 0.3903f;
    }

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
    bridgeInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

    return checkIntervalOverlap(innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax);
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool matchRadiiBBBEE23478(TAcc const& acc,
                                                           float innerRadius,
                                                           float bridgeRadius,
                                                           float outerRadius,
                                                           float bridgeRadiusMin2S,
                                                           float bridgeRadiusMax2S) {
    float innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax;

    float innerInvRadiusErrorBound = 0.2097f;
    float bridgeInvRadiusErrorBound = 0.8557f;

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
    bridgeInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

    return checkIntervalOverlap(innerInvRadiusMin,
                                innerInvRadiusMax,
                                alpaka::math::min(acc, bridgeInvRadiusMin, 1.0f / bridgeRadiusMax2S),
                                alpaka::math::max(acc, bridgeInvRadiusMax, 1.0f / bridgeRadiusMin2S));
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool matchRadiiBBBEE34578(TAcc const& acc,
                                                           float innerRadius,
                                                           float bridgeRadius,
                                                           float outerRadius,
                                                           float bridgeRadiusMin2S,
                                                           float bridgeRadiusMax2S) {
    float innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax;

    float innerInvRadiusErrorBound = 0.066f;
    float bridgeInvRadiusErrorBound = 0.617f;

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
    bridgeInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

    return checkIntervalOverlap(innerInvRadiusMin,
                                innerInvRadiusMax,
                                alpaka::math::min(acc, bridgeInvRadiusMin, 1.0f / bridgeRadiusMax2S),
                                alpaka::math::max(acc, bridgeInvRadiusMax, 1.0f / bridgeRadiusMin2S));
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool matchRadiiBBEEE(TAcc const& acc,
                                                      float innerRadius,
                                                      float bridgeRadius,
                                                      float outerRadius,
                                                      float bridgeRadiusMin2S,
                                                      float bridgeRadiusMax2S) {
    float innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax;

    float innerInvRadiusErrorBound = 0.6376f;
    float bridgeInvRadiusErrorBound = 2.1381f;

    if (innerRadius > 2.0f / (2.f * k2Rinv1GeVf))  //as good as no selections!
    {
      innerInvRadiusErrorBound = 12.9173f;
      bridgeInvRadiusErrorBound = 5.1700f;
    }

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
    bridgeInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

    return checkIntervalOverlap(innerInvRadiusMin,
                                innerInvRadiusMax,
                                alpaka::math::min(acc, bridgeInvRadiusMin, 1.0f / bridgeRadiusMax2S),
                                alpaka::math::max(acc, bridgeInvRadiusMax, 1.0f / bridgeRadiusMin2S));
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool matchRadiiBEEEE(TAcc const& acc,
                                                      float innerRadius,
                                                      float bridgeRadius,
                                                      float outerRadius,
                                                      float innerRadiusMin2S,
                                                      float innerRadiusMax2S,
                                                      float bridgeRadiusMin2S,
                                                      float bridgeRadiusMax2S) {
    float innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax;

    float innerInvRadiusErrorBound = 1.9382f;
    float bridgeInvRadiusErrorBound = 3.7280f;

    if (innerRadius > 2.0f / (2.f * k2Rinv1GeVf)) {
      innerInvRadiusErrorBound = 23.2713f;
      bridgeInvRadiusErrorBound = 21.7980f;
    }

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
    bridgeInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

    return checkIntervalOverlap(alpaka::math::min(acc, innerInvRadiusMin, 1.0 / innerRadiusMax2S),
                                alpaka::math::max(acc, innerInvRadiusMax, 1.0 / innerRadiusMin2S),
                                alpaka::math::min(acc, bridgeInvRadiusMin, 1.0 / bridgeRadiusMax2S),
                                alpaka::math::max(acc, bridgeInvRadiusMax, 1.0 / bridgeRadiusMin2S));
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool matchRadiiEEEEE(TAcc const& acc,
                                                      float innerRadius,
                                                      float bridgeRadius,
                                                      float outerRadius,
                                                      float innerRadiusMin2S,
                                                      float innerRadiusMax2S,
                                                      float bridgeRadiusMin2S,
                                                      float bridgeRadiusMax2S) {
    float innerInvRadiusMin, innerInvRadiusMax, bridgeInvRadiusMin, bridgeInvRadiusMax;

    float innerInvRadiusErrorBound = 1.9382f;
    float bridgeInvRadiusErrorBound = 2.2091f;

    if (innerRadius > 2.0f / (2.f * k2Rinv1GeVf)) {
      innerInvRadiusErrorBound = 22.5226f;
      bridgeInvRadiusErrorBound = 21.0966f;
    }

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    bridgeInvRadiusMax = (1.f + bridgeInvRadiusErrorBound) / bridgeRadius;
    bridgeInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - bridgeInvRadiusErrorBound) / bridgeRadius);

    return checkIntervalOverlap(alpaka::math::min(acc, innerInvRadiusMin, 1.0 / innerRadiusMax2S),
                                alpaka::math::max(acc, innerInvRadiusMax, 1.0 / innerRadiusMin2S),
                                alpaka::math::min(acc, bridgeInvRadiusMin, 1.0 / bridgeRadiusMax2S),
                                alpaka::math::max(acc, bridgeInvRadiusMax, 1.0 / bridgeRadiusMin2S));
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void computeSigmasForRegression(TAcc const& acc,
                                                                 Modules const& modulesInGPU,
                                                                 const uint16_t* lowerModuleIndices,
                                                                 float* delta1,
                                                                 float* delta2,
                                                                 float* slopes,
                                                                 bool* isFlat,
                                                                 unsigned int nPoints = 5,
                                                                 bool anchorHits = true) {
    /*
        Bool anchorHits required to deal with a weird edge case wherein 
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
      } else {
#ifdef WARNINGS
        printf("ERROR!!!!! I SHOULDN'T BE HERE!!!! subdet = %d, type = %d, side = %d\n",
               moduleSubdet,
               moduleType,
               moduleSide);
#endif
      }
    }
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computeRadiusUsingRegression(TAcc const& acc,
                                                                    unsigned int nPoints,
                                                                    float* xs,
                                                                    float* ys,
                                                                    float* delta1,
                                                                    float* delta2,
                                                                    float* slopes,
                                                                    bool* isFlat,
                                                                    float& g,
                                                                    float& f,
                                                                    float* sigmas2,
                                                                    float& chiSquared) {
    float radius = 0.f;

    // Some extra variables
    // the two variables will be called x1 and x2, and y (which is x^2 + y^2)

    float sigmaX1Squared = 0.f;
    float sigmaX2Squared = 0.f;
    float sigmaX1X2 = 0.f;
    float sigmaX1y = 0.f;
    float sigmaX2y = 0.f;
    float sigmaY = 0.f;
    float sigmaX1 = 0.f;
    float sigmaX2 = 0.f;
    float sigmaOne = 0.f;

    float xPrime, yPrime, absArctanSlope, angleM;
    for (size_t i = 0; i < nPoints; i++) {
      // Computing sigmas is a very tricky affair
      // if the module is tilted or endcap, we need to use the slopes properly!

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
      sigmas2[i] = 4 * ((xPrime * delta1[i]) * (xPrime * delta1[i]) + (yPrime * delta2[i]) * (yPrime * delta2[i]));

      sigmaX1Squared += (xs[i] * xs[i]) / sigmas2[i];
      sigmaX2Squared += (ys[i] * ys[i]) / sigmas2[i];
      sigmaX1X2 += (xs[i] * ys[i]) / sigmas2[i];
      sigmaX1y += (xs[i] * (xs[i] * xs[i] + ys[i] * ys[i])) / sigmas2[i];
      sigmaX2y += (ys[i] * (xs[i] * xs[i] + ys[i] * ys[i])) / sigmas2[i];
      sigmaY += (xs[i] * xs[i] + ys[i] * ys[i]) / sigmas2[i];
      sigmaX1 += xs[i] / sigmas2[i];
      sigmaX2 += ys[i] / sigmas2[i];
      sigmaOne += 1.0f / sigmas2[i];
    }
    float denominator = (sigmaX1X2 - sigmaX1 * sigmaX2) * (sigmaX1X2 - sigmaX1 * sigmaX2) -
                        (sigmaX1Squared - sigmaX1 * sigmaX1) * (sigmaX2Squared - sigmaX2 * sigmaX2);

    float twoG = ((sigmaX2y - sigmaX2 * sigmaY) * (sigmaX1X2 - sigmaX1 * sigmaX2) -
                  (sigmaX1y - sigmaX1 * sigmaY) * (sigmaX2Squared - sigmaX2 * sigmaX2)) /
                 denominator;
    float twoF = ((sigmaX1y - sigmaX1 * sigmaY) * (sigmaX1X2 - sigmaX1 * sigmaX2) -
                  (sigmaX2y - sigmaX2 * sigmaY) * (sigmaX1Squared - sigmaX1 * sigmaX1)) /
                 denominator;

    float c = -(sigmaY - twoG * sigmaX1 - twoF * sigmaX2) / sigmaOne;
    g = 0.5f * twoG;
    f = 0.5f * twoF;
    if (g * g + f * f - c < 0) {
#ifdef WARNINGS
      printf("FATAL! r^2 < 0!\n");
#endif
      chiSquared = -1;
      return -1;
    }

    radius = alpaka::math::sqrt(acc, g * g + f * f - c);
    // compute chi squared
    chiSquared = 0.f;
    for (size_t i = 0; i < nPoints; i++) {
      chiSquared += (xs[i] * xs[i] + ys[i] * ys[i] - twoG * xs[i] - twoF * ys[i] + c) *
                    (xs[i] * xs[i] + ys[i] * ys[i] - twoG * xs[i] - twoF * ys[i] + c) / sigmas2[i];
    }
    return radius;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computeChiSquared(TAcc const& acc,
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
    // given values of (g, f, radius) and a set of points (and its uncertainties)
    // compute chi squared
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
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void runDeltaBetaIterationsT5(TAcc const& acc,
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
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletDefaultAlgoBBBB(TAcc const& acc,
                                                                   Modules const& modulesInGPU,
                                                                   MiniDoublets const& mdsInGPU,
                                                                   Segments const& segmentsInGPU,
                                                                   uint16_t innerInnerLowerModuleIndex,
                                                                   uint16_t innerOuterLowerModuleIndex,
                                                                   uint16_t outerInnerLowerModuleIndex,
                                                                   uint16_t outerOuterLowerModuleIndex,
                                                                   unsigned int innerSegmentIndex,
                                                                   unsigned int outerSegmentIndex,
                                                                   unsigned int firstMDIndex,
                                                                   unsigned int secondMDIndex,
                                                                   unsigned int thirdMDIndex,
                                                                   unsigned int fourthMDIndex) {
    bool isPS_InLo = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == PS);
    bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == PS);

    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InOut = mdsInGPU.anchorRt[secondMDIndex];
    float rt_OutLo = mdsInGPU.anchorRt[thirdMDIndex];

    float z_InLo = mdsInGPU.anchorZ[firstMDIndex];
    float z_InOut = mdsInGPU.anchorZ[secondMDIndex];
    float z_OutLo = mdsInGPU.anchorZ[thirdMDIndex];

    float alpha1GeV_OutLo =
        alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / ptCut, kSinAlphaMax));

    float rtRatio_OutLoInLo = rt_OutLo / rt_InLo;  // Outer segment beginning rt divided by inner segment beginning rt;
    float dzDrtScale =
        alpaka::math::tan(acc, alpha1GeV_OutLo) / alpha1GeV_OutLo;  // The track can bend in r-z plane slightly
    float zpitch_InLo = (isPS_InLo ? kPixelPSZpitch : kStrip2SZpitch);
    float zpitch_OutLo = (isPS_OutLo ? kPixelPSZpitch : kStrip2SZpitch);

    float zHi = z_InLo + (z_InLo + kDeltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo < 0.f ? 1.f : dzDrtScale) +
                (zpitch_InLo + zpitch_OutLo);
    float zLo = z_InLo + (z_InLo - kDeltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo > 0.f ? 1.f : dzDrtScale) -
                (zpitch_InLo + zpitch_OutLo);

    //Cut 1 - z compatibility
    if ((z_OutLo < zLo) || (z_OutLo > zHi))
      return false;

    float drt_OutLo_InLo = (rt_OutLo - rt_InLo);
    float r3_InLo = alpaka::math::sqrt(acc, z_InLo * z_InLo + rt_InLo * rt_InLo);
    float drt_InSeg = rt_InOut - rt_InLo;
    float dz_InSeg = z_InOut - z_InLo;
    float dr3_InSeg = alpaka::math::sqrt(acc, rt_InOut * rt_InOut + z_InOut * z_InOut) -
                      alpaka::math::sqrt(acc, rt_InLo * rt_InLo + z_InLo * z_InLo);

    float coshEta = dr3_InSeg / drt_InSeg;
    float dzErr = (zpitch_InLo + zpitch_OutLo) * (zpitch_InLo + zpitch_OutLo) * 2.f;

    float thetaMuls2 = (kMulsInGeV * kMulsInGeV) * (0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f) * (r3_InLo / rt_InLo);
    float muls2 = thetaMuls2 * 9.f / (ptCut * ptCut) * 16.f;
    dzErr += muls2 * drt_OutLo_InLo * drt_OutLo_InLo / 3.f * coshEta * coshEta;
    dzErr = alpaka::math::sqrt(acc, dzErr);

    // Constructing upper and lower bound
    const float dzMean = dz_InSeg / drt_InSeg * drt_OutLo_InLo;
    const float zWindow =
        dzErr / drt_InSeg * drt_OutLo_InLo +
        (zpitch_InLo + zpitch_OutLo);  //FIXME for ptCut lower than ~0.8 need to add curv path correction
    float zLoPointed = z_InLo + dzMean * (z_InLo > 0.f ? 1.f : dzDrtScale) - zWindow;
    float zHiPointed = z_InLo + dzMean * (z_InLo < 0.f ? 1.f : dzDrtScale) + zWindow;

    // Cut #2: Pointed Z (Inner segment two MD points to outer segment inner MD)
    if ((z_OutLo < zLoPointed) || (z_OutLo > zHiPointed))
      return false;

    float pvOffset = 0.1f / rt_OutLo;
    float dPhiCut = alpha1GeV_OutLo + alpaka::math::sqrt(acc, muls2 + pvOffset * pvOffset);

    float deltaPhiPos = phi_mpi_pi(acc, mdsInGPU.anchorPhi[fourthMDIndex] - mdsInGPU.anchorPhi[secondMDIndex]);
    // Cut #3: FIXME:deltaPhiPos can be tighter
    if (alpaka::math::abs(acc, deltaPhiPos) > dPhiCut)
      return false;

    float midPointX = 0.5f * (mdsInGPU.anchorX[firstMDIndex] + mdsInGPU.anchorX[thirdMDIndex]);
    float midPointY = 0.5f * (mdsInGPU.anchorY[firstMDIndex] + mdsInGPU.anchorY[thirdMDIndex]);
    float diffX = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float diffY = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];

    float dPhi = deltaPhi(acc, midPointX, midPointY, diffX, diffY);

    // Cut #4: deltaPhiChange
    if (alpaka::math::abs(acc, dPhi) > dPhiCut)
      return false;

    // First obtaining the raw betaIn and betaOut values without any correction and just purely based on the mini-doublet hit positions

    float alpha_InLo = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
    float alpha_OutLo = __H2F(segmentsInGPU.dPhiChanges[outerSegmentIndex]);

    bool isEC_lastLayer = modulesInGPU.subdets[outerOuterLowerModuleIndex] == Endcap and
                          modulesInGPU.moduleType[outerOuterLowerModuleIndex] == TwoS;

    float alpha_OutUp, alpha_OutUp_highEdge, alpha_OutUp_lowEdge;

    alpha_OutUp = phi_mpi_pi(acc,
                             phi(acc,
                                 mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex],
                                 mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) -
                                 mdsInGPU.anchorPhi[fourthMDIndex]);

    alpha_OutUp_highEdge = alpha_OutUp;
    alpha_OutUp_lowEdge = alpha_OutUp;

    float tl_axis_x = mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float tl_axis_y = mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];
    float tl_axis_highEdge_x = tl_axis_x;
    float tl_axis_highEdge_y = tl_axis_y;
    float tl_axis_lowEdge_x = tl_axis_x;
    float tl_axis_lowEdge_y = tl_axis_y;

    float betaIn = alpha_InLo - phi_mpi_pi(acc, phi(acc, tl_axis_x, tl_axis_y) - mdsInGPU.anchorPhi[firstMDIndex]);

    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;
    float betaOut = -alpha_OutUp + phi_mpi_pi(acc, phi(acc, tl_axis_x, tl_axis_y) - mdsInGPU.anchorPhi[fourthMDIndex]);

    float betaOutRHmin = betaOut;
    float betaOutRHmax = betaOut;

    if (isEC_lastLayer) {
      alpha_OutUp_highEdge = phi_mpi_pi(acc,
                                        phi(acc,
                                            mdsInGPU.anchorHighEdgeX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex],
                                            mdsInGPU.anchorHighEdgeY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) -
                                            mdsInGPU.anchorHighEdgePhi[fourthMDIndex]);
      alpha_OutUp_lowEdge = phi_mpi_pi(acc,
                                       phi(acc,
                                           mdsInGPU.anchorLowEdgeX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex],
                                           mdsInGPU.anchorLowEdgeY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) -
                                           mdsInGPU.anchorLowEdgePhi[fourthMDIndex]);

      tl_axis_highEdge_x = mdsInGPU.anchorHighEdgeX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
      tl_axis_highEdge_y = mdsInGPU.anchorHighEdgeY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];
      tl_axis_lowEdge_x = mdsInGPU.anchorLowEdgeX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
      tl_axis_lowEdge_y = mdsInGPU.anchorLowEdgeY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];

      betaOutRHmin =
          -alpha_OutUp_highEdge +
          phi_mpi_pi(acc, phi(acc, tl_axis_highEdge_x, tl_axis_highEdge_y) - mdsInGPU.anchorHighEdgePhi[fourthMDIndex]);
      betaOutRHmax =
          -alpha_OutUp_lowEdge +
          phi_mpi_pi(acc, phi(acc, tl_axis_lowEdge_x, tl_axis_lowEdge_y) - mdsInGPU.anchorLowEdgePhi[fourthMDIndex]);
    }

    //beta computation
    float drt_tl_axis = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);

    float corrF = 1.f;
    //innerOuterAnchor - innerInnerAnchor
    const float rt_InSeg =
        alpaka::math::sqrt(acc,
                           (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) *
                                   (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) +
                               (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]) *
                                   (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]));
    float betaInCut =
        alpaka::math::asin(
            acc, alpaka::math::min(acc, (-rt_InSeg * corrF + drt_tl_axis) * k2Rinv1GeVf / ptCut, kSinAlphaMax)) +
        (0.02f / drt_InSeg);

    //Cut #5: first beta cut
    if (alpaka::math::abs(acc, betaInRHmin) >= betaInCut)
      return false;

    float betaAv = 0.5f * (betaIn + betaOut);
    float pt_beta = drt_tl_axis * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);
    int lIn = 5;
    int lOut = isEC_lastLayer ? 11 : 5;
    float sdOut_dr = alpaka::math::sqrt(acc,
                                        (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) *
                                                (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) +
                                            (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) *
                                                (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]));
    float sdOut_d = mdsInGPU.anchorRt[fourthMDIndex] - mdsInGPU.anchorRt[thirdMDIndex];

    runDeltaBetaIterationsT5(acc, betaIn, betaOut, betaAv, pt_beta, rt_InSeg, sdOut_dr, drt_tl_axis, lIn);

    const float betaInMMSF = (alpaka::math::abs(acc, betaInRHmin + betaInRHmax) > 0)
                                 ? (2.f * betaIn / alpaka::math::abs(acc, betaInRHmin + betaInRHmax))
                                 : 0.f;  //mean value of min,max is the old betaIn
    const float betaOutMMSF = (alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax) > 0)
                                  ? (2.f * betaOut / alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax))
                                  : 0.f;
    betaInRHmin *= betaInMMSF;
    betaInRHmax *= betaInMMSF;
    betaOutRHmin *= betaOutMMSF;
    betaOutRHmax *= betaOutMMSF;

    float min_ptBeta_maxPtBeta = alpaka::math::min(
        acc, alpaka::math::abs(acc, pt_beta), kPt_betaMax);  //need to confimm the range-out value of 7 GeV
    const float dBetaMuls2 = thetaMuls2 * 16.f / (min_ptBeta_maxPtBeta * min_ptBeta_maxPtBeta);

    const float alphaInAbsReg =
        alpaka::math::max(acc,
                          alpaka::math::abs(acc, alpha_InLo),
                          alpaka::math::asin(acc, alpaka::math::min(acc, rt_InLo * k2Rinv1GeVf / 3.0f, kSinAlphaMax)));
    const float alphaOutAbsReg =
        alpaka::math::max(acc,
                          alpaka::math::abs(acc, alpha_OutLo),
                          alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / 3.0f, kSinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : alpaka::math::abs(acc, alphaInAbsReg * kDeltaZLum / z_InLo);
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

    float betaOutCut =
        alpaka::math::asin(acc, alpaka::math::min(acc, drt_tl_axis * k2Rinv1GeVf / ptCut, kSinAlphaMax)) +
        (0.02f / sdOut_d) + alpaka::math::sqrt(acc, dBetaLum2 + dBetaMuls2);

    //Cut #6: The real beta cut
    if (alpaka::math::abs(acc, betaOut) >= betaOutCut)
      return false;

    float dBetaRes = 0.02f / alpaka::math::min(acc, sdOut_d, drt_InSeg);
    float dBetaCut2 =
        (dBetaRes * dBetaRes * 2.0f + dBetaMuls2 + dBetaLum2 + dBetaRIn2 + dBetaROut2 +
         0.25f *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));

    float dBeta = betaIn - betaOut;
    return dBeta * dBeta <= dBetaCut2;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletDefaultAlgoBBEE(TAcc const& acc,
                                                                   Modules const& modulesInGPU,
                                                                   MiniDoublets const& mdsInGPU,
                                                                   Segments const& segmentsInGPU,
                                                                   uint16_t innerInnerLowerModuleIndex,
                                                                   uint16_t innerOuterLowerModuleIndex,
                                                                   uint16_t outerInnerLowerModuleIndex,
                                                                   uint16_t outerOuterLowerModuleIndex,
                                                                   unsigned int innerSegmentIndex,
                                                                   unsigned int outerSegmentIndex,
                                                                   unsigned int firstMDIndex,
                                                                   unsigned int secondMDIndex,
                                                                   unsigned int thirdMDIndex,
                                                                   unsigned int fourthMDIndex) {
    bool isPS_InLo = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == PS);
    bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == PS);

    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InOut = mdsInGPU.anchorRt[secondMDIndex];
    float rt_OutLo = mdsInGPU.anchorRt[thirdMDIndex];

    float z_InLo = mdsInGPU.anchorZ[firstMDIndex];
    float z_InOut = mdsInGPU.anchorZ[secondMDIndex];
    float z_OutLo = mdsInGPU.anchorZ[thirdMDIndex];

    float alpha1GeV_OutLo =
        alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / ptCut, kSinAlphaMax));

    float dzDrtScale =
        alpaka::math::tan(acc, alpha1GeV_OutLo) / alpha1GeV_OutLo;  // The track can bend in r-z plane slightly
    float zpitch_InLo = (isPS_InLo ? kPixelPSZpitch : kStrip2SZpitch);
    float zpitch_OutLo = (isPS_OutLo ? kPixelPSZpitch : kStrip2SZpitch);
    float zGeom = zpitch_InLo + zpitch_OutLo;

    // Cut #0: Preliminary (Only here in endcap case)
    if (z_InLo * z_OutLo <= 0)
      return false;

    float dLum = alpaka::math::copysign(acc, kDeltaZLum, z_InLo);
    bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerInnerLowerModuleIndex] == PS;
    float rtGeom1 = isOutSgInnerMDPS ? kPixelPSZpitch : kStrip2SZpitch;
    float zGeom1 = alpaka::math::copysign(acc, zGeom, z_InLo);
    float rtLo = rt_InLo * (1.f + (z_OutLo - z_InLo - zGeom1) / (z_InLo + zGeom1 + dLum) / dzDrtScale) -
                 rtGeom1;  //slope correction only on the lower end
    float rtOut = rt_OutLo;

    //Cut #1: rt condition
    if (rtOut < rtLo)
      return false;

    float zInForHi = z_InLo - zGeom1 - dLum;
    if (zInForHi * z_InLo < 0) {
      zInForHi = alpaka::math::copysign(acc, 0.1f, z_InLo);
    }
    float rtHi = rt_InLo * (1.f + (z_OutLo - z_InLo + zGeom1) / zInForHi) + rtGeom1;

    //Cut #2: rt condition
    if ((rt_OutLo < rtLo) || (rt_OutLo > rtHi))
      return false;

    float rIn = alpaka::math::sqrt(acc, z_InLo * z_InLo + rt_InLo * rt_InLo);
    const float drtSDIn = rt_InOut - rt_InLo;
    const float dzSDIn = z_InOut - z_InLo;
    const float dr3SDIn = alpaka::math::sqrt(acc, rt_InOut * rt_InOut + z_InOut * z_InOut) -
                          alpaka::math::sqrt(acc, rt_InLo * rt_InLo + z_InLo * z_InLo);

    const float coshEta = dr3SDIn / drtSDIn;  //direction estimate
    const float dzOutInAbs = alpaka::math::abs(acc, z_OutLo - z_InLo);
    const float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);
    const float zGeom1_another = kPixelPSZpitch;
    float kZ = (z_OutLo - z_InLo) / dzSDIn;
    float drtErr =
        zGeom1_another * zGeom1_another * drtSDIn * drtSDIn / dzSDIn / dzSDIn * (1.f - 2.f * kZ + 2.f * kZ * kZ);
    const float thetaMuls2 = (kMulsInGeV * kMulsInGeV) * (0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f) * (rIn / rt_InLo);
    const float muls2 = thetaMuls2 * 9.f / (ptCut * ptCut) * 16.f;
    drtErr += muls2 * multDzDr * multDzDr / 3.f * coshEta * coshEta;
    drtErr = alpaka::math::sqrt(acc, drtErr);

    //Cut #3: rt-z pointed
    if ((kZ < 0) || (rtOut < rtLo) || (rtOut > rtHi))
      return false;

    const float pvOffset = 0.1f / rt_OutLo;
    float dPhiCut = alpha1GeV_OutLo + alpaka::math::sqrt(acc, muls2 + pvOffset * pvOffset);

    float deltaPhiPos = phi_mpi_pi(acc, mdsInGPU.anchorPhi[fourthMDIndex] - mdsInGPU.anchorPhi[secondMDIndex]);

    //Cut #4: deltaPhiPos can be tighter
    if (alpaka::math::abs(acc, deltaPhiPos) > dPhiCut)
      return false;

    float midPointX = 0.5f * (mdsInGPU.anchorX[firstMDIndex] + mdsInGPU.anchorX[thirdMDIndex]);
    float midPointY = 0.5f * (mdsInGPU.anchorY[firstMDIndex] + mdsInGPU.anchorY[thirdMDIndex]);
    float diffX = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float diffY = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];

    float dPhi = deltaPhi(acc, midPointX, midPointY, diffX, diffY);
    // Cut #5: deltaPhiChange
    if (alpaka::math::abs(acc, dPhi) > dPhiCut)
      return false;

    float sdIn_alpha = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
    float sdIn_alpha_min = __H2F(segmentsInGPU.dPhiChangeMins[innerSegmentIndex]);
    float sdIn_alpha_max = __H2F(segmentsInGPU.dPhiChangeMaxs[innerSegmentIndex]);
    float sdOut_alpha = sdIn_alpha;

    float sdOut_alphaOut = phi_mpi_pi(acc,
                                      phi(acc,
                                          mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex],
                                          mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) -
                                          mdsInGPU.anchorPhi[fourthMDIndex]);

    float sdOut_alphaOut_min = phi_mpi_pi(
        acc, __H2F(segmentsInGPU.dPhiChangeMins[outerSegmentIndex]) - __H2F(segmentsInGPU.dPhiMins[outerSegmentIndex]));
    float sdOut_alphaOut_max = phi_mpi_pi(
        acc, __H2F(segmentsInGPU.dPhiChangeMaxs[outerSegmentIndex]) - __H2F(segmentsInGPU.dPhiMaxs[outerSegmentIndex]));

    float tl_axis_x = mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float tl_axis_y = mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];

    float betaIn = sdIn_alpha - phi_mpi_pi(acc, phi(acc, tl_axis_x, tl_axis_y) - mdsInGPU.anchorPhi[firstMDIndex]);

    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;
    float betaOut =
        -sdOut_alphaOut + phi_mpi_pi(acc, phi(acc, tl_axis_x, tl_axis_y) - mdsInGPU.anchorPhi[fourthMDIndex]);

    float betaOutRHmin = betaOut;
    float betaOutRHmax = betaOut;

    bool isEC_secondLayer = (modulesInGPU.subdets[innerOuterLowerModuleIndex] == Endcap) and
                            (modulesInGPU.moduleType[innerOuterLowerModuleIndex] == TwoS);

    if (isEC_secondLayer) {
      betaInRHmin = betaIn - sdIn_alpha_min + sdIn_alpha;
      betaInRHmax = betaIn - sdIn_alpha_max + sdIn_alpha;
    }

    betaOutRHmin = betaOut - sdOut_alphaOut_min + sdOut_alphaOut;
    betaOutRHmax = betaOut - sdOut_alphaOut_max + sdOut_alphaOut;

    float swapTemp;
    if (alpaka::math::abs(acc, betaOutRHmin) > alpaka::math::abs(acc, betaOutRHmax)) {
      swapTemp = betaOutRHmin;
      betaOutRHmin = betaOutRHmax;
      betaOutRHmax = swapTemp;
    }

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
    const float corrF = 1.f;
    float betaInCut =
        alpaka::math::asin(acc, alpaka::math::min(acc, (-sdIn_dr * corrF + dr) * k2Rinv1GeVf / ptCut, kSinAlphaMax)) +
        (0.02f / sdIn_d);

    //Cut #6: first beta cut
    if (alpaka::math::abs(acc, betaInRHmin) >= betaInCut)
      return false;

    float betaAv = 0.5f * (betaIn + betaOut);
    float pt_beta = dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);

    float lIn = 5;
    float lOut = 11;

    float sdOut_dr = alpaka::math::sqrt(acc,
                                        (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) *
                                                (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) +
                                            (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) *
                                                (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]));
    float sdOut_d = mdsInGPU.anchorRt[fourthMDIndex] - mdsInGPU.anchorRt[thirdMDIndex];

    runDeltaBetaIterationsT5(acc, betaIn, betaOut, betaAv, pt_beta, sdIn_dr, sdOut_dr, dr, lIn);

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

    float min_ptBeta_maxPtBeta = alpaka::math::min(
        acc, alpaka::math::abs(acc, pt_beta), kPt_betaMax);  //need to confirm the range-out value of 7 GeV
    const float dBetaMuls2 = thetaMuls2 * 16.f / (min_ptBeta_maxPtBeta * min_ptBeta_maxPtBeta);

    const float alphaInAbsReg =
        alpaka::math::max(acc,
                          alpaka::math::abs(acc, sdIn_alpha),
                          alpaka::math::asin(acc, alpaka::math::min(acc, rt_InLo * k2Rinv1GeVf / 3.0f, kSinAlphaMax)));
    const float alphaOutAbsReg =
        alpaka::math::max(acc,
                          alpaka::math::abs(acc, sdOut_alpha),
                          alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / 3.0f, kSinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : alpaka::math::abs(acc, alphaInAbsReg * kDeltaZLum / z_InLo);
    const float dBetaOutLum = lOut < 11 ? 0.0f : alpaka::math::abs(acc, alphaOutAbsReg * kDeltaZLum / z_OutLo);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);
    const float sinDPhi = alpaka::math::sin(acc, dPhi);

    const float dBetaRIn2 = 0;  // TODO-RH
    float dBetaROut = 0;
    if (modulesInGPU.moduleType[outerOuterLowerModuleIndex] == TwoS) {
      dBetaROut =
          (alpaka::math::sqrt(acc,
                              mdsInGPU.anchorHighEdgeX[fourthMDIndex] * mdsInGPU.anchorHighEdgeX[fourthMDIndex] +
                                  mdsInGPU.anchorHighEdgeY[fourthMDIndex] * mdsInGPU.anchorHighEdgeY[fourthMDIndex]) -
           alpaka::math::sqrt(acc,
                              mdsInGPU.anchorLowEdgeX[fourthMDIndex] * mdsInGPU.anchorLowEdgeX[fourthMDIndex] +
                                  mdsInGPU.anchorLowEdgeY[fourthMDIndex] * mdsInGPU.anchorLowEdgeY[fourthMDIndex])) *
          sinDPhi / dr;
    }

    const float dBetaROut2 = dBetaROut * dBetaROut;
    float betaOutCut = alpaka::math::asin(acc, alpaka::math::min(acc, dr * k2Rinv1GeVf / ptCut, kSinAlphaMax)) +
                       (0.02f / sdOut_d) + alpaka::math::sqrt(acc, dBetaLum2 + dBetaMuls2);

    //Cut #6: The real beta cut
    if (alpaka::math::abs(acc, betaOut) >= betaOutCut)
      return false;

    float dBetaRes = 0.02f / alpaka::math::min(acc, sdOut_d, sdIn_d);
    float dBetaCut2 =
        (dBetaRes * dBetaRes * 2.0f + dBetaMuls2 + dBetaLum2 + dBetaRIn2 + dBetaROut2 +
         0.25f *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));
    float dBeta = betaIn - betaOut;
    //Cut #7: Cut on dBet
    return dBeta * dBeta <= dBetaCut2;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletDefaultAlgoEEEE(TAcc const& acc,
                                                                   Modules const& modulesInGPU,
                                                                   MiniDoublets const& mdsInGPU,
                                                                   Segments const& segmentsInGPU,
                                                                   uint16_t innerInnerLowerModuleIndex,
                                                                   uint16_t innerOuterLowerModuleIndex,
                                                                   uint16_t outerInnerLowerModuleIndex,
                                                                   uint16_t outerOuterLowerModuleIndex,
                                                                   unsigned int innerSegmentIndex,
                                                                   unsigned int outerSegmentIndex,
                                                                   unsigned int firstMDIndex,
                                                                   unsigned int secondMDIndex,
                                                                   unsigned int thirdMDIndex,
                                                                   unsigned int fourthMDIndex) {
    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InOut = mdsInGPU.anchorRt[secondMDIndex];
    float rt_OutLo = mdsInGPU.anchorRt[thirdMDIndex];

    float z_InLo = mdsInGPU.anchorZ[firstMDIndex];
    float z_InOut = mdsInGPU.anchorZ[secondMDIndex];
    float z_OutLo = mdsInGPU.anchorZ[thirdMDIndex];

    float alpha1GeV_OutLo =
        alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / ptCut, kSinAlphaMax));

    float dzDrtScale =
        alpaka::math::tan(acc, alpha1GeV_OutLo) / alpha1GeV_OutLo;  // The track can bend in r-z plane slightly

    // Cut #0: Preliminary (Only here in endcap case)
    if ((z_InLo * z_OutLo) <= 0)
      return false;

    float dLum = alpaka::math::copysign(acc, kDeltaZLum, z_InLo);
    bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerInnerLowerModuleIndex] == PS;
    bool isInSgInnerMDPS = modulesInGPU.moduleType[innerInnerLowerModuleIndex] == PS;

    float rtGeom = (isInSgInnerMDPS and isOutSgInnerMDPS)  ? 2.f * kPixelPSZpitch
                   : (isInSgInnerMDPS or isOutSgInnerMDPS) ? kPixelPSZpitch + kStrip2SZpitch
                                                           : 2.f * kStrip2SZpitch;

    float dz = z_OutLo - z_InLo;
    float rtLo = rt_InLo * (1.f + dz / (z_InLo + dLum) / dzDrtScale) - rtGeom;  //slope correction only on the lower end

    float rtOut = rt_OutLo;

    //Cut #1: rt condition

    float rtHi = rt_InLo * (1.f + dz / (z_InLo - dLum)) + rtGeom;

    if ((rtOut < rtLo) || (rtOut > rtHi))
      return false;

    bool isInSgOuterMDPS = modulesInGPU.moduleType[innerOuterLowerModuleIndex] == PS;

    const float drtSDIn = rt_InOut - rt_InLo;
    const float dzSDIn = z_InOut - z_InLo;
    const float dr3SDIn = alpaka::math::sqrt(acc, rt_InOut * rt_InOut + z_InOut * z_InOut) -
                          alpaka::math::sqrt(acc, rt_InLo * rt_InLo + z_InLo * z_InLo);
    float coshEta = dr3SDIn / drtSDIn;  //direction estimate
    float dzOutInAbs = alpaka::math::abs(acc, z_OutLo - z_InLo);
    float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);

    float kZ = (z_OutLo - z_InLo) / dzSDIn;
    float thetaMuls2 = (kMulsInGeV * kMulsInGeV) * (0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f);

    float muls2 = thetaMuls2 * 9.f / (ptCut * ptCut) * 16.f;

    float drtErr =
        alpaka::math::sqrt(acc,
                           kPixelPSZpitch * kPixelPSZpitch * 2.f / (dzSDIn * dzSDIn) * (dzOutInAbs * dzOutInAbs) +
                               muls2 * multDzDr * multDzDr / 3.f * coshEta * coshEta);

    float drtMean = drtSDIn * dzOutInAbs / alpaka::math::abs(acc, dzSDIn);
    float rtWindow = drtErr + rtGeom;
    float rtLo_point = rt_InLo + drtMean / dzDrtScale - rtWindow;
    float rtHi_point = rt_InLo + drtMean + rtWindow;

    // Cut #3: rt-z pointed
    // https://github.com/slava77/cms-tkph2-ntuple/blob/superDoubletLinked-91X-noMock/doubletAnalysis.C#L3765

    if (isInSgInnerMDPS and isInSgOuterMDPS)  // If both PS then we can point
    {
      if (kZ < 0 || rtOut < rtLo_point || rtOut > rtHi_point)
        return false;
    }

    float pvOffset = 0.1f / rtOut;
    float dPhiCut = alpha1GeV_OutLo + alpaka::math::sqrt(acc, muls2 + pvOffset * pvOffset);

    float deltaPhiPos = phi_mpi_pi(acc, mdsInGPU.anchorPhi[fourthMDIndex] - mdsInGPU.anchorPhi[secondMDIndex]);

    if (alpaka::math::abs(acc, deltaPhiPos) > dPhiCut)
      return false;

    float midPointX = 0.5f * (mdsInGPU.anchorX[firstMDIndex] + mdsInGPU.anchorX[thirdMDIndex]);
    float midPointY = 0.5f * (mdsInGPU.anchorY[firstMDIndex] + mdsInGPU.anchorY[thirdMDIndex]);
    float diffX = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float diffY = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];

    float dPhi = deltaPhi(acc, midPointX, midPointY, diffX, diffY);

    // Cut #5: deltaPhiChange
    if (alpaka::math::abs(acc, dPhi) > dPhiCut)
      return false;

    float sdIn_alpha = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
    float sdOut_alpha = sdIn_alpha;  //weird
    float sdOut_dPhiPos = phi_mpi_pi(acc, mdsInGPU.anchorPhi[fourthMDIndex] - mdsInGPU.anchorPhi[thirdMDIndex]);

    float sdOut_dPhiChange = __H2F(segmentsInGPU.dPhiChanges[outerSegmentIndex]);
    float sdOut_dPhiChange_min = __H2F(segmentsInGPU.dPhiChangeMins[outerSegmentIndex]);
    float sdOut_dPhiChange_max = __H2F(segmentsInGPU.dPhiChangeMaxs[outerSegmentIndex]);

    float sdOut_alphaOutRHmin = phi_mpi_pi(acc, sdOut_dPhiChange_min - sdOut_dPhiPos);
    float sdOut_alphaOutRHmax = phi_mpi_pi(acc, sdOut_dPhiChange_max - sdOut_dPhiPos);
    float sdOut_alphaOut = phi_mpi_pi(acc, sdOut_dPhiChange - sdOut_dPhiPos);

    float tl_axis_x = mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float tl_axis_y = mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];

    float betaIn = sdIn_alpha - phi_mpi_pi(acc, phi(acc, tl_axis_x, tl_axis_y) - mdsInGPU.anchorPhi[firstMDIndex]);

    float sdIn_alphaRHmin = __H2F(segmentsInGPU.dPhiChangeMins[innerSegmentIndex]);
    float sdIn_alphaRHmax = __H2F(segmentsInGPU.dPhiChangeMaxs[innerSegmentIndex]);
    float betaInRHmin = betaIn + sdIn_alphaRHmin - sdIn_alpha;
    float betaInRHmax = betaIn + sdIn_alphaRHmax - sdIn_alpha;

    float betaOut =
        -sdOut_alphaOut + phi_mpi_pi(acc, phi(acc, tl_axis_x, tl_axis_y) - mdsInGPU.anchorPhi[fourthMDIndex]);

    float betaOutRHmin = betaOut - sdOut_alphaOutRHmin + sdOut_alphaOut;
    float betaOutRHmax = betaOut - sdOut_alphaOutRHmax + sdOut_alphaOut;

    float swapTemp;
    if (alpaka::math::abs(acc, betaOutRHmin) > alpaka::math::abs(acc, betaOutRHmax)) {
      swapTemp = betaOutRHmin;
      betaOutRHmin = betaOutRHmax;
      betaOutRHmax = swapTemp;
    }

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
    const float corrF = 1.f;
    float betaInCut =
        alpaka::math::asin(acc, alpaka::math::min(acc, (-sdIn_dr * corrF + dr) * k2Rinv1GeVf / ptCut, kSinAlphaMax)) +
        (0.02f / sdIn_d);

    //Cut #6: first beta cut
    if (alpaka::math::abs(acc, betaInRHmin) >= betaInCut)
      return false;

    float betaAv = 0.5f * (betaIn + betaOut);
    float pt_beta = dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);

    int lIn = 11;   //endcap
    int lOut = 13;  //endcap

    float sdOut_dr = alpaka::math::sqrt(acc,
                                        (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) *
                                                (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) +
                                            (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) *
                                                (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]));
    float sdOut_d = mdsInGPU.anchorRt[fourthMDIndex] - mdsInGPU.anchorRt[thirdMDIndex];

    runDeltaBetaIterationsT5(acc, betaIn, betaOut, betaAv, pt_beta, sdIn_dr, sdOut_dr, dr, lIn);

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

    float min_ptBeta_maxPtBeta = alpaka::math::min(
        acc, alpaka::math::abs(acc, pt_beta), kPt_betaMax);  //need to confirm the range-out value of 7 GeV
    const float dBetaMuls2 = thetaMuls2 * 16.f / (min_ptBeta_maxPtBeta * min_ptBeta_maxPtBeta);

    const float alphaInAbsReg =
        alpaka::math::max(acc,
                          alpaka::math::abs(acc, sdIn_alpha),
                          alpaka::math::asin(acc, alpaka::math::min(acc, rt_InLo * k2Rinv1GeVf / 3.0f, kSinAlphaMax)));
    const float alphaOutAbsReg =
        alpaka::math::max(acc,
                          alpaka::math::abs(acc, sdOut_alpha),
                          alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / 3.0f, kSinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : alpaka::math::abs(acc, alphaInAbsReg * kDeltaZLum / z_InLo);
    const float dBetaOutLum = lOut < 11 ? 0.0f : alpaka::math::abs(acc, alphaOutAbsReg * kDeltaZLum / z_OutLo);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);

    const float dBetaRIn2 = 0;  // TODO-RH

    float dBetaROut2 = 0;  //TODO-RH
    float betaOutCut = alpaka::math::asin(acc, alpaka::math::min(acc, dr * k2Rinv1GeVf / ptCut, kSinAlphaMax)) +
                       (0.02f / sdOut_d) + alpaka::math::sqrt(acc, dBetaLum2 + dBetaMuls2);

    //Cut #6: The real beta cut
    if (alpaka::math::abs(acc, betaOut) >= betaOutCut)
      return false;

    float dBetaRes = 0.02f / alpaka::math::min(acc, sdOut_d, sdIn_d);
    float dBetaCut2 =
        (dBetaRes * dBetaRes * 2.0f + dBetaMuls2 + dBetaLum2 + dBetaRIn2 + dBetaROut2 +
         0.25f *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));
    float dBeta = betaIn - betaOut;
    //Cut #7: Cut on dBeta
    return dBeta * dBeta <= dBetaCut2;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletAlgoSelector(TAcc const& acc,
                                                                Modules const& modulesInGPU,
                                                                MiniDoublets const& mdsInGPU,
                                                                Segments const& segmentsInGPU,
                                                                uint16_t innerInnerLowerModuleIndex,
                                                                uint16_t innerOuterLowerModuleIndex,
                                                                uint16_t outerInnerLowerModuleIndex,
                                                                uint16_t outerOuterLowerModuleIndex,
                                                                unsigned int innerSegmentIndex,
                                                                unsigned int outerSegmentIndex,
                                                                unsigned int firstMDIndex,
                                                                unsigned int secondMDIndex,
                                                                unsigned int thirdMDIndex,
                                                                unsigned int fourthMDIndex) {
    short innerInnerLowerModuleSubdet = modulesInGPU.subdets[innerInnerLowerModuleIndex];
    short innerOuterLowerModuleSubdet = modulesInGPU.subdets[innerOuterLowerModuleIndex];
    short outerInnerLowerModuleSubdet = modulesInGPU.subdets[outerInnerLowerModuleIndex];
    short outerOuterLowerModuleSubdet = modulesInGPU.subdets[outerOuterLowerModuleIndex];

    if (innerInnerLowerModuleSubdet == Barrel and innerOuterLowerModuleSubdet == Barrel and
        outerInnerLowerModuleSubdet == Barrel and outerOuterLowerModuleSubdet == Barrel) {
      return runQuintupletDefaultAlgoBBBB(acc,
                                          modulesInGPU,
                                          mdsInGPU,
                                          segmentsInGPU,
                                          innerInnerLowerModuleIndex,
                                          innerOuterLowerModuleIndex,
                                          outerInnerLowerModuleIndex,
                                          outerOuterLowerModuleIndex,
                                          innerSegmentIndex,
                                          outerSegmentIndex,
                                          firstMDIndex,
                                          secondMDIndex,
                                          thirdMDIndex,
                                          fourthMDIndex);
    } else if (innerInnerLowerModuleSubdet == Barrel and innerOuterLowerModuleSubdet == Barrel and
               outerInnerLowerModuleSubdet == Endcap and outerOuterLowerModuleSubdet == Endcap) {
      return runQuintupletDefaultAlgoBBEE(acc,
                                          modulesInGPU,
                                          mdsInGPU,
                                          segmentsInGPU,
                                          innerInnerLowerModuleIndex,
                                          innerOuterLowerModuleIndex,
                                          outerInnerLowerModuleIndex,
                                          outerOuterLowerModuleIndex,
                                          innerSegmentIndex,
                                          outerSegmentIndex,
                                          firstMDIndex,
                                          secondMDIndex,
                                          thirdMDIndex,
                                          fourthMDIndex);
    } else if (innerInnerLowerModuleSubdet == Barrel and innerOuterLowerModuleSubdet == Barrel and
               outerInnerLowerModuleSubdet == Barrel and outerOuterLowerModuleSubdet == Endcap) {
      return runQuintupletDefaultAlgoBBBB(acc,
                                          modulesInGPU,
                                          mdsInGPU,
                                          segmentsInGPU,
                                          innerInnerLowerModuleIndex,
                                          innerOuterLowerModuleIndex,
                                          outerInnerLowerModuleIndex,
                                          outerOuterLowerModuleIndex,
                                          innerSegmentIndex,
                                          outerSegmentIndex,
                                          firstMDIndex,
                                          secondMDIndex,
                                          thirdMDIndex,
                                          fourthMDIndex);
    } else if (innerInnerLowerModuleSubdet == Barrel and innerOuterLowerModuleSubdet == Endcap and
               outerInnerLowerModuleSubdet == Endcap and outerOuterLowerModuleSubdet == Endcap) {
      return runQuintupletDefaultAlgoBBEE(acc,
                                          modulesInGPU,
                                          mdsInGPU,
                                          segmentsInGPU,
                                          innerInnerLowerModuleIndex,
                                          innerOuterLowerModuleIndex,
                                          outerInnerLowerModuleIndex,
                                          outerOuterLowerModuleIndex,
                                          innerSegmentIndex,
                                          outerSegmentIndex,
                                          firstMDIndex,
                                          secondMDIndex,
                                          thirdMDIndex,
                                          fourthMDIndex);
    } else if (innerInnerLowerModuleSubdet == Endcap and innerOuterLowerModuleSubdet == Endcap and
               outerInnerLowerModuleSubdet == Endcap and outerOuterLowerModuleSubdet == Endcap) {
      return runQuintupletDefaultAlgoEEEE(acc,
                                          modulesInGPU,
                                          mdsInGPU,
                                          segmentsInGPU,
                                          innerInnerLowerModuleIndex,
                                          innerOuterLowerModuleIndex,
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

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletDefaultAlgo(TAcc const& acc,
                                                               Modules& modulesInGPU,
                                                               MiniDoublets& mdsInGPU,
                                                               Segments& segmentsInGPU,
                                                               Triplets& tripletsInGPU,
                                                               uint16_t lowerModuleIndex1,
                                                               uint16_t lowerModuleIndex2,
                                                               uint16_t lowerModuleIndex3,
                                                               uint16_t lowerModuleIndex4,
                                                               uint16_t lowerModuleIndex5,
                                                               unsigned int innerTripletIndex,
                                                               unsigned int outerTripletIndex,
                                                               float& innerRadius,
                                                               float& outerRadius,
                                                               float& bridgeRadius,
                                                               float& regressionG,
                                                               float& regressionF,
                                                               float& regressionRadius,
                                                               float& rzChiSquared,
                                                               float& chiSquared,
                                                               float& nonAnchorChiSquared,
                                                               bool& TightCutFlag) {
    unsigned int firstSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex];
    unsigned int secondSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex + 1];
    unsigned int thirdSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex];
    unsigned int fourthSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex + 1];

    unsigned int innerOuterOuterMiniDoubletIndex =
        segmentsInGPU.mdIndices[2 * secondSegmentIndex + 1];  //inner triplet outer segment outer MD index
    unsigned int outerInnerInnerMiniDoubletIndex =
        segmentsInGPU.mdIndices[2 * thirdSegmentIndex];  //outer triplet inner segment inner MD index

    //this cut reduces the number of candidates by a factor of 3, i.e., 2 out of 3 warps can end right here!
    if (innerOuterOuterMiniDoubletIndex != outerInnerInnerMiniDoubletIndex)
      return false;

    unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 * firstSegmentIndex];
    unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * secondSegmentIndex];
    unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * secondSegmentIndex + 1];
    unsigned int fourthMDIndex = segmentsInGPU.mdIndices[2 * thirdSegmentIndex + 1];
    unsigned int fifthMDIndex = segmentsInGPU.mdIndices[2 * fourthSegmentIndex + 1];

    if (not runQuintupletAlgoSelector(acc,
                                      modulesInGPU,
                                      mdsInGPU,
                                      segmentsInGPU,
                                      lowerModuleIndex1,
                                      lowerModuleIndex2,
                                      lowerModuleIndex3,
                                      lowerModuleIndex4,
                                      firstSegmentIndex,
                                      thirdSegmentIndex,
                                      firstMDIndex,
                                      secondMDIndex,
                                      thirdMDIndex,
                                      fourthMDIndex))
      return false;

    if (not runQuintupletAlgoSelector(acc,
                                      modulesInGPU,
                                      mdsInGPU,
                                      segmentsInGPU,
                                      lowerModuleIndex1,
                                      lowerModuleIndex2,
                                      lowerModuleIndex4,
                                      lowerModuleIndex5,
                                      firstSegmentIndex,
                                      fourthSegmentIndex,
                                      firstMDIndex,
                                      secondMDIndex,
                                      fourthMDIndex,
                                      fifthMDIndex))
      return false;

    float x1 = mdsInGPU.anchorX[firstMDIndex];
    float x2 = mdsInGPU.anchorX[secondMDIndex];
    float x3 = mdsInGPU.anchorX[thirdMDIndex];
    float x4 = mdsInGPU.anchorX[fourthMDIndex];
    float x5 = mdsInGPU.anchorX[fifthMDIndex];

    float y1 = mdsInGPU.anchorY[firstMDIndex];
    float y2 = mdsInGPU.anchorY[secondMDIndex];
    float y3 = mdsInGPU.anchorY[thirdMDIndex];
    float y4 = mdsInGPU.anchorY[fourthMDIndex];
    float y5 = mdsInGPU.anchorY[fifthMDIndex];

    //construct the arrays
    float x1Vec[] = {x1, x1, x1};
    float y1Vec[] = {y1, y1, y1};
    float x2Vec[] = {x2, x2, x2};
    float y2Vec[] = {y2, y2, y2};
    float x3Vec[] = {x3, x3, x3};
    float y3Vec[] = {y3, y3, y3};

    if (modulesInGPU.subdets[lowerModuleIndex1] == Endcap and modulesInGPU.moduleType[lowerModuleIndex1] == TwoS) {
      x1Vec[1] = mdsInGPU.anchorLowEdgeX[firstMDIndex];
      x1Vec[2] = mdsInGPU.anchorHighEdgeX[firstMDIndex];

      y1Vec[1] = mdsInGPU.anchorLowEdgeY[firstMDIndex];
      y1Vec[2] = mdsInGPU.anchorHighEdgeY[firstMDIndex];
    }
    if (modulesInGPU.subdets[lowerModuleIndex2] == Endcap and modulesInGPU.moduleType[lowerModuleIndex2] == TwoS) {
      x2Vec[1] = mdsInGPU.anchorLowEdgeX[secondMDIndex];
      x2Vec[2] = mdsInGPU.anchorHighEdgeX[secondMDIndex];

      y2Vec[1] = mdsInGPU.anchorLowEdgeY[secondMDIndex];
      y2Vec[2] = mdsInGPU.anchorHighEdgeY[secondMDIndex];
    }
    if (modulesInGPU.subdets[lowerModuleIndex3] == Endcap and modulesInGPU.moduleType[lowerModuleIndex3] == TwoS) {
      x3Vec[1] = mdsInGPU.anchorLowEdgeX[thirdMDIndex];
      x3Vec[2] = mdsInGPU.anchorHighEdgeX[thirdMDIndex];

      y3Vec[1] = mdsInGPU.anchorLowEdgeY[thirdMDIndex];
      y3Vec[2] = mdsInGPU.anchorHighEdgeY[thirdMDIndex];
    }

    float innerRadiusMin2S, innerRadiusMax2S;
    computeErrorInRadius(acc, x1Vec, y1Vec, x2Vec, y2Vec, x3Vec, y3Vec, innerRadiusMin2S, innerRadiusMax2S);

    for (int i = 0; i < 3; i++) {
      x1Vec[i] = x4;
      y1Vec[i] = y4;
    }
    if (modulesInGPU.subdets[lowerModuleIndex4] == Endcap and modulesInGPU.moduleType[lowerModuleIndex4] == TwoS) {
      x1Vec[1] = mdsInGPU.anchorLowEdgeX[fourthMDIndex];
      x1Vec[2] = mdsInGPU.anchorHighEdgeX[fourthMDIndex];

      y1Vec[1] = mdsInGPU.anchorLowEdgeY[fourthMDIndex];
      y1Vec[2] = mdsInGPU.anchorHighEdgeY[fourthMDIndex];
    }

    float bridgeRadiusMin2S, bridgeRadiusMax2S;
    computeErrorInRadius(acc, x2Vec, y2Vec, x3Vec, y3Vec, x1Vec, y1Vec, bridgeRadiusMin2S, bridgeRadiusMax2S);

    for (int i = 0; i < 3; i++) {
      x2Vec[i] = x5;
      y2Vec[i] = y5;
    }
    if (modulesInGPU.subdets[lowerModuleIndex5] == Endcap and modulesInGPU.moduleType[lowerModuleIndex5] == TwoS) {
      x2Vec[1] = mdsInGPU.anchorLowEdgeX[fifthMDIndex];
      x2Vec[2] = mdsInGPU.anchorHighEdgeX[fifthMDIndex];

      y2Vec[1] = mdsInGPU.anchorLowEdgeY[fifthMDIndex];
      y2Vec[2] = mdsInGPU.anchorHighEdgeY[fifthMDIndex];
    }

    float outerRadiusMin2S, outerRadiusMax2S;
    computeErrorInRadius(acc, x3Vec, y3Vec, x1Vec, y1Vec, x2Vec, y2Vec, outerRadiusMin2S, outerRadiusMax2S);

    float g, f;
    outerRadius = tripletsInGPU.circleRadius[outerTripletIndex];
    bridgeRadius = computeRadiusFromThreeAnchorHits(acc, x2, y2, x3, y3, x4, y4, g, f);
    innerRadius = tripletsInGPU.circleRadius[innerTripletIndex];
    g = tripletsInGPU.circleCenterX[innerTripletIndex];
    f = tripletsInGPU.circleCenterY[innerTripletIndex];

#ifdef USE_RZCHI2
    float inner_pt = 2 * k2Rinv1GeVf * innerRadius;

    if (not passT5RZConstraint(acc,
                               modulesInGPU,
                               mdsInGPU,
                               firstMDIndex,
                               secondMDIndex,
                               thirdMDIndex,
                               fourthMDIndex,
                               fifthMDIndex,
                               lowerModuleIndex1,
                               lowerModuleIndex2,
                               lowerModuleIndex3,
                               lowerModuleIndex4,
                               lowerModuleIndex5,
                               rzChiSquared,
                               inner_pt,
                               innerRadius,
                               g,
                               f,
                               TightCutFlag))
      return false;
#else
    rzChiSquared = -1;
#endif
    if (innerRadius < 0.95f * ptCut / (2.f * k2Rinv1GeVf))
      return false;

    //split by category
    bool matchedRadii;
    if (modulesInGPU.subdets[lowerModuleIndex1] == Barrel and modulesInGPU.subdets[lowerModuleIndex2] == Barrel and
        modulesInGPU.subdets[lowerModuleIndex3] == Barrel and modulesInGPU.subdets[lowerModuleIndex4] == Barrel and
        modulesInGPU.subdets[lowerModuleIndex5] == Barrel) {
      matchedRadii = matchRadiiBBBBB(acc, innerRadius, bridgeRadius, outerRadius);
    } else if (modulesInGPU.subdets[lowerModuleIndex1] == Barrel and
               modulesInGPU.subdets[lowerModuleIndex2] == Barrel and
               modulesInGPU.subdets[lowerModuleIndex3] == Barrel and
               modulesInGPU.subdets[lowerModuleIndex4] == Barrel and
               modulesInGPU.subdets[lowerModuleIndex5] == Endcap) {
      matchedRadii = matchRadiiBBBBE(acc, innerRadius, bridgeRadius, outerRadius);
    } else if (modulesInGPU.subdets[lowerModuleIndex1] == Barrel and
               modulesInGPU.subdets[lowerModuleIndex2] == Barrel and
               modulesInGPU.subdets[lowerModuleIndex3] == Barrel and
               modulesInGPU.subdets[lowerModuleIndex4] == Endcap and
               modulesInGPU.subdets[lowerModuleIndex5] == Endcap) {
      if (modulesInGPU.layers[lowerModuleIndex1] == 1) {
        matchedRadii =
            matchRadiiBBBEE12378(acc, innerRadius, bridgeRadius, outerRadius, bridgeRadiusMin2S, bridgeRadiusMax2S);
      } else if (modulesInGPU.layers[lowerModuleIndex1] == 2) {
        matchedRadii =
            matchRadiiBBBEE23478(acc, innerRadius, bridgeRadius, outerRadius, bridgeRadiusMin2S, bridgeRadiusMax2S);
      } else {
        matchedRadii =
            matchRadiiBBBEE34578(acc, innerRadius, bridgeRadius, outerRadius, bridgeRadiusMin2S, bridgeRadiusMax2S);
      }
    }

    else if (modulesInGPU.subdets[lowerModuleIndex1] == Barrel and modulesInGPU.subdets[lowerModuleIndex2] == Barrel and
             modulesInGPU.subdets[lowerModuleIndex3] == Endcap and modulesInGPU.subdets[lowerModuleIndex4] == Endcap and
             modulesInGPU.subdets[lowerModuleIndex5] == Endcap) {
      matchedRadii = matchRadiiBBEEE(acc, innerRadius, bridgeRadius, outerRadius, bridgeRadiusMin2S, bridgeRadiusMax2S);
    } else if (modulesInGPU.subdets[lowerModuleIndex1] == Barrel and
               modulesInGPU.subdets[lowerModuleIndex2] == Endcap and
               modulesInGPU.subdets[lowerModuleIndex3] == Endcap and
               modulesInGPU.subdets[lowerModuleIndex4] == Endcap and
               modulesInGPU.subdets[lowerModuleIndex5] == Endcap) {
      matchedRadii = matchRadiiBEEEE(acc,
                                     innerRadius,
                                     bridgeRadius,
                                     outerRadius,
                                     innerRadiusMin2S,
                                     innerRadiusMax2S,
                                     bridgeRadiusMin2S,
                                     bridgeRadiusMax2S);
    } else {
      matchedRadii = matchRadiiEEEEE(acc,
                                     innerRadius,
                                     bridgeRadius,
                                     outerRadius,
                                     innerRadiusMin2S,
                                     innerRadiusMax2S,
                                     bridgeRadiusMin2S,
                                     bridgeRadiusMax2S);
    }

    //compute regression radius right here - this computation is expensive!!!
    if (not matchedRadii)
      return false;

    float xVec[] = {x1, x2, x3, x4, x5};
    float yVec[] = {y1, y2, y3, y4, y5};
    const uint16_t lowerModuleIndices[] = {
        lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5};

    // 5 categories for sigmas
    float sigmas2[5], delta1[5], delta2[5], slopes[5];
    bool isFlat[5];

    computeSigmasForRegression(acc, modulesInGPU, lowerModuleIndices, delta1, delta2, slopes, isFlat);
    regressionRadius = computeRadiusUsingRegression(acc,
                                                    Params_T5::kLayers,
                                                    xVec,
                                                    yVec,
                                                    delta1,
                                                    delta2,
                                                    slopes,
                                                    isFlat,
                                                    regressionG,
                                                    regressionF,
                                                    sigmas2,
                                                    chiSquared);

#ifdef USE_T5_DNN
    unsigned int mdIndices[] = {firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, fifthMDIndex};
    float inference = t5dnn::runInference(acc,
                                          modulesInGPU,
                                          mdsInGPU,
                                          segmentsInGPU,
                                          tripletsInGPU,
                                          xVec,
                                          yVec,
                                          mdIndices,
                                          lowerModuleIndices,
                                          innerTripletIndex,
                                          outerTripletIndex,
                                          innerRadius,
                                          outerRadius,
                                          bridgeRadius);
    TightCutFlag = TightCutFlag and (inference > t5dnn::kLSTWp2);  // T5-in-TC cut
    if (inference <= t5dnn::kLSTWp2)                               // T5-building cut
      return false;
#endif

#ifdef USE_RPHICHI2
    // extra chi squared cuts!
    if (regressionRadius < 5.0f / (2.f * k2Rinv1GeVf)) {
      if (not passChiSquaredConstraint(modulesInGPU,
                                       lowerModuleIndex1,
                                       lowerModuleIndex2,
                                       lowerModuleIndex3,
                                       lowerModuleIndex4,
                                       lowerModuleIndex5,
                                       chiSquared))
        return false;
    }
#endif

    //compute the other chisquared
    //non anchor is always shifted for tilted and endcap!
    float nonAnchorDelta1[Params_T5::kLayers], nonAnchorDelta2[Params_T5::kLayers], nonAnchorSlopes[Params_T5::kLayers];
    float nonAnchorxs[] = {mdsInGPU.outerX[firstMDIndex],
                           mdsInGPU.outerX[secondMDIndex],
                           mdsInGPU.outerX[thirdMDIndex],
                           mdsInGPU.outerX[fourthMDIndex],
                           mdsInGPU.outerX[fifthMDIndex]};
    float nonAnchorys[] = {mdsInGPU.outerY[firstMDIndex],
                           mdsInGPU.outerY[secondMDIndex],
                           mdsInGPU.outerY[thirdMDIndex],
                           mdsInGPU.outerY[fourthMDIndex],
                           mdsInGPU.outerY[fifthMDIndex]};

    computeSigmasForRegression(acc,
                               modulesInGPU,
                               lowerModuleIndices,
                               nonAnchorDelta1,
                               nonAnchorDelta2,
                               nonAnchorSlopes,
                               isFlat,
                               Params_T5::kLayers,
                               false);
    nonAnchorChiSquared = computeChiSquared(acc,
                                            Params_T5::kLayers,
                                            nonAnchorxs,
                                            nonAnchorys,
                                            nonAnchorDelta1,
                                            nonAnchorDelta2,
                                            nonAnchorSlopes,
                                            isFlat,
                                            regressionG,
                                            regressionF,
                                            regressionRadius);
    return true;
  }

  struct CreateQuintupletsInGPUv2 {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  Modules modulesInGPU,
                                  MiniDoublets mdsInGPU,
                                  Segments segmentsInGPU,
                                  Triplets tripletsInGPU,
                                  Quintuplets quintupletsInGPU,
                                  ObjectRanges rangesInGPU,
                                  uint16_t nEligibleT5Modules) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (int iter = globalThreadIdx[0]; iter < nEligibleT5Modules; iter += gridThreadExtent[0]) {
        uint16_t lowerModule1 = rangesInGPU.indicesOfEligibleT5Modules[iter];
        short layer2_adjustment;
        int layer = modulesInGPU.layers[lowerModule1];
        if (layer == 1) {
          layer2_adjustment = 1;
        }  // get upper segment to be in second layer
        else if (layer == 2) {
          layer2_adjustment = 0;
        }  // get lower segment to be in second layer
        else {
          continue;
        }
        unsigned int nInnerTriplets = tripletsInGPU.nTriplets[lowerModule1];
        for (unsigned int innerTripletArrayIndex = globalThreadIdx[1]; innerTripletArrayIndex < nInnerTriplets;
             innerTripletArrayIndex += gridThreadExtent[1]) {
          unsigned int innerTripletIndex = rangesInGPU.tripletModuleIndices[lowerModule1] + innerTripletArrayIndex;
          uint16_t lowerModule2 = tripletsInGPU.lowerModuleIndices[Params_T3::kLayers * innerTripletIndex + 1];
          uint16_t lowerModule3 = tripletsInGPU.lowerModuleIndices[Params_T3::kLayers * innerTripletIndex + 2];
          unsigned int nOuterTriplets = tripletsInGPU.nTriplets[lowerModule3];
          for (unsigned int outerTripletArrayIndex = globalThreadIdx[2]; outerTripletArrayIndex < nOuterTriplets;
               outerTripletArrayIndex += gridThreadExtent[2]) {
            unsigned int outerTripletIndex = rangesInGPU.tripletModuleIndices[lowerModule3] + outerTripletArrayIndex;
            uint16_t lowerModule4 = tripletsInGPU.lowerModuleIndices[Params_T3::kLayers * outerTripletIndex + 1];
            uint16_t lowerModule5 = tripletsInGPU.lowerModuleIndices[Params_T3::kLayers * outerTripletIndex + 2];

            float innerRadius, outerRadius, bridgeRadius, regressionG, regressionF, regressionRadius, rzChiSquared,
                chiSquared, nonAnchorChiSquared;  //required for making distributions

            bool TightCutFlag = false;
            bool success = runQuintupletDefaultAlgo(acc,
                                                    modulesInGPU,
                                                    mdsInGPU,
                                                    segmentsInGPU,
                                                    tripletsInGPU,
                                                    lowerModule1,
                                                    lowerModule2,
                                                    lowerModule3,
                                                    lowerModule4,
                                                    lowerModule5,
                                                    innerTripletIndex,
                                                    outerTripletIndex,
                                                    innerRadius,
                                                    outerRadius,
                                                    bridgeRadius,
                                                    regressionG,
                                                    regressionF,
                                                    regressionRadius,
                                                    rzChiSquared,
                                                    chiSquared,
                                                    nonAnchorChiSquared,
                                                    TightCutFlag);

            if (success) {
              int totOccupancyQuintuplets = alpaka::atomicAdd(
                  acc, &quintupletsInGPU.totOccupancyQuintuplets[lowerModule1], 1u, alpaka::hierarchy::Threads{});
              if (totOccupancyQuintuplets >= rangesInGPU.quintupletModuleOccupancy[lowerModule1]) {
#ifdef WARNINGS
                printf("Quintuplet excess alert! Module index = %d\n", lowerModule1);
#endif
              } else {
                int quintupletModuleIndex = alpaka::atomicAdd(
                    acc, &quintupletsInGPU.nQuintuplets[lowerModule1], 1u, alpaka::hierarchy::Threads{});
                //this if statement should never get executed!
                if (rangesInGPU.quintupletModuleIndices[lowerModule1] == -1) {
#ifdef WARNINGS
                  printf("Quintuplets : no memory for module at module index = %d\n", lowerModule1);
#endif
                } else {
                  unsigned int quintupletIndex =
                      rangesInGPU.quintupletModuleIndices[lowerModule1] + quintupletModuleIndex;
                  float phi =
                      mdsInGPU.anchorPhi[segmentsInGPU.mdIndices[2 * tripletsInGPU.segmentIndices[2 * innerTripletIndex +
                                                                                                  layer2_adjustment]]];
                  float eta =
                      mdsInGPU.anchorEta[segmentsInGPU.mdIndices[2 * tripletsInGPU.segmentIndices[2 * innerTripletIndex +
                                                                                                  layer2_adjustment]]];
                  float pt = (innerRadius + outerRadius) * k2Rinv1GeVf;
                  float scores = chiSquared + nonAnchorChiSquared;
                  addQuintupletToMemory(tripletsInGPU,
                                        quintupletsInGPU,
                                        innerTripletIndex,
                                        outerTripletIndex,
                                        lowerModule1,
                                        lowerModule2,
                                        lowerModule3,
                                        lowerModule4,
                                        lowerModule5,
                                        innerRadius,
                                        bridgeRadius,
                                        outerRadius,
                                        regressionG,
                                        regressionF,
                                        regressionRadius,
                                        rzChiSquared,
                                        chiSquared,
                                        nonAnchorChiSquared,
                                        pt,
                                        eta,
                                        phi,
                                        scores,
                                        layer,
                                        quintupletIndex,
                                        TightCutFlag);

                  tripletsInGPU.partOfT5[quintupletsInGPU.tripletIndices[2 * quintupletIndex]] = true;
                  tripletsInGPU.partOfT5[quintupletsInGPU.tripletIndices[2 * quintupletIndex + 1]] = true;
                }
              }
            }
          }
        }
      }
    }
  };

  struct CreateEligibleModulesListForQuintupletsGPU {
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

      // Initialize variables in shared memory and set to 0
      int& nEligibleT5Modulesx = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      int& nTotalQuintupletsx = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        nTotalQuintupletsx = 0;
        nEligibleT5Modulesx = 0;
      }
      alpaka::syncBlockThreads(acc);

      // Create variables outside of the for loop.
      int occupancy, category_number, eta_number;

      for (int i = globalThreadIdx[0]; i < *modulesInGPU.nLowerModules; i += gridThreadExtent[0]) {
        // Condition for a quintuple to exist for a module
        // TCs don't exist for layers 5 and 6 barrel, and layers 2,3,4,5 endcap
        short module_rings = modulesInGPU.rings[i];
        short module_layers = modulesInGPU.layers[i];
        short module_subdets = modulesInGPU.subdets[i];
        float module_eta = alpaka::math::abs(acc, modulesInGPU.eta[i]);

        if (tripletsInGPU.nTriplets[i] == 0)
          continue;
        if (module_subdets == Barrel and module_layers >= 3)
          continue;
        if (module_subdets == Endcap and module_layers > 1)
          continue;

        int nEligibleT5Modules = alpaka::atomicAdd(acc, &nEligibleT5Modulesx, 1, alpaka::hierarchy::Threads{});

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
          occupancy = 336;
        else if (category_number == 0 && eta_number == 1)
          occupancy = 414;
        else if (category_number == 0 && eta_number == 2)
          occupancy = 231;
        else if (category_number == 0 && eta_number == 3)
          occupancy = 146;
        else if (category_number == 3 && eta_number == 1)
          occupancy = 0;
        else if (category_number == 3 && eta_number == 2)
          occupancy = 191;
        else if (category_number == 3 && eta_number == 3)
          occupancy = 106;
        else {
          occupancy = 0;
#ifdef WARNINGS
          printf("Unhandled case in createEligibleModulesListForQuintupletsGPU! Module index = %i\n", i);
#endif
        }

        int nTotQ = alpaka::atomicAdd(acc, &nTotalQuintupletsx, occupancy, alpaka::hierarchy::Threads{});
        rangesInGPU.quintupletModuleIndices[i] = nTotQ;
        rangesInGPU.indicesOfEligibleT5Modules[nEligibleT5Modules] = i;
        rangesInGPU.quintupletModuleOccupancy[i] = occupancy;
      }

      // Wait for all threads to finish before reporting final values
      alpaka::syncBlockThreads(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        *rangesInGPU.nEligibleT5Modules = static_cast<uint16_t>(nEligibleT5Modulesx);
        *rangesInGPU.device_nTotalQuints = static_cast<unsigned int>(nTotalQuintupletsx);
      }
    }
  };

  struct AddQuintupletRangesToEventExplicit {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  Modules modulesInGPU,
                                  Quintuplets quintupletsInGPU,
                                  ObjectRanges rangesInGPU) const {
      // implementation is 1D with a single block
      static_assert(std::is_same_v<TAcc, ALPAKA_ACCELERATOR_NAMESPACE::Acc1D>, "Should be Acc1D");
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (uint16_t i = globalThreadIdx[0]; i < *modulesInGPU.nLowerModules; i += gridThreadExtent[0]) {
        if (quintupletsInGPU.nQuintuplets[i] == 0 or rangesInGPU.quintupletModuleIndices[i] == -1) {
          rangesInGPU.quintupletRanges[i * 2] = -1;
          rangesInGPU.quintupletRanges[i * 2 + 1] = -1;
        } else {
          rangesInGPU.quintupletRanges[i * 2] = rangesInGPU.quintupletModuleIndices[i];
          rangesInGPU.quintupletRanges[i * 2 + 1] =
              rangesInGPU.quintupletModuleIndices[i] + quintupletsInGPU.nQuintuplets[i] - 1;
        }
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
