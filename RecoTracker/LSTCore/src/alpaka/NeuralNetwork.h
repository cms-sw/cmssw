#ifndef NeuralNetwork_cuh
#define NeuralNetwork_cuh

#ifdef LST_IS_CMSSW_PACKAGE
#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/alpaka/Module.h"
#else
#include "Constants.h"
#include "Module.h"
#endif

#include "NeuralNetworkWeights.h"
#include "Segment.h"
#include "MiniDoublet.h"
#include "Hit.h"
#include "Triplet.h"

namespace T5DNN {

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float runInference(TAcc const& acc,
                                                    struct SDL::modules& modulesInGPU,
                                                    struct SDL::miniDoublets& mdsInGPU,
                                                    struct SDL::segments& segmentsInGPU,
                                                    struct SDL::triplets& tripletsInGPU,
                                                    const float* xVec,
                                                    const float* yVec,
                                                    const unsigned int* mdIndices,
                                                    const uint16_t* lowerModuleIndices,
                                                    const unsigned int& innerTripletIndex,
                                                    const unsigned int& outerTripletIndex,
                                                    const float& innerRadius,
                                                    const float& outerRadius,
                                                    const float& bridgeRadius) {
    // Unpack x-coordinates of hits
    float x1 = xVec[0];
    float x2 = xVec[1];
    float x3 = xVec[2];
    float x4 = xVec[3];
    float x5 = xVec[4];
    // Unpack y-coordinates of hits
    float y1 = yVec[0];
    float y2 = yVec[1];
    float y3 = yVec[2];
    float y4 = yVec[3];
    float y5 = yVec[4];
    // Unpack module indices
    unsigned int mdIndex1 = mdIndices[0];
    unsigned int mdIndex2 = mdIndices[1];
    unsigned int mdIndex3 = mdIndices[2];
    unsigned int mdIndex4 = mdIndices[3];
    unsigned int mdIndex5 = mdIndices[4];
    // Unpack module indices
    uint16_t lowerModuleIndex1 = lowerModuleIndices[0];
    uint16_t lowerModuleIndex2 = lowerModuleIndices[1];
    uint16_t lowerModuleIndex3 = lowerModuleIndices[2];
    uint16_t lowerModuleIndex4 = lowerModuleIndices[3];
    uint16_t lowerModuleIndex5 = lowerModuleIndices[4];

    // Compute some convenience variables
    short layer2_adjustment = 0;
    if (modulesInGPU.layers[lowerModuleIndex1] == 1) {
      layer2_adjustment = 1;  // get upper segment to be in second layer
    }
    unsigned int md_idx_for_t5_eta_phi =
        segmentsInGPU.mdIndices[2 * tripletsInGPU.segmentIndices[2 * innerTripletIndex + layer2_adjustment]];
    bool is_endcap1 = (modulesInGPU.subdets[lowerModuleIndex1] == 4);  // true if anchor hit 1 is in the endcap
    bool is_endcap2 = (modulesInGPU.subdets[lowerModuleIndex2] == 4);  // true if anchor hit 2 is in the endcap
    bool is_endcap3 = (modulesInGPU.subdets[lowerModuleIndex3] == 4);  // true if anchor hit 3 is in the endcap
    bool is_endcap4 = (modulesInGPU.subdets[lowerModuleIndex4] == 4);  // true if anchor hit 4 is in the endcap
    bool is_endcap5 = (modulesInGPU.subdets[lowerModuleIndex5] == 4);  // true if anchor hit 5 is in the endcap

    // Build DNN input vector (corresponding output N-tuple branch noted in parenthetical in comment)
    float x[38] = {
        SDL::temp_log10(acc, 2 * SDL::k2Rinv1GeVf * innerRadius),        // inner T3 pT (t3_pt)
        mdsInGPU.anchorEta[mdIndex1],                                    // inner T3 anchor hit 1 eta (t3_0_eta)
        mdsInGPU.anchorPhi[mdIndex1],                                    // inner T3 anchor hit 1 phi (t3_0_phi)
        mdsInGPU.anchorZ[mdIndex1],                                      // inner T3 anchor hit 1 z (t3_0_z)
        alpaka::math::sqrt(acc, x1 * x1 + y1 * y1),                      // inner T3 anchor hit 1 r (t3_0_r)
        float(modulesInGPU.layers[lowerModuleIndex1] + 6 * is_endcap1),  // inner T3 anchor hit 1 layer (t3_0_layer)
        mdsInGPU.anchorEta[mdIndex2],                                    // inner T3 anchor hit 2 eta (t3_2_eta)
        mdsInGPU.anchorPhi[mdIndex2],                                    // inner T3 anchor hit 2 phi (t3_2_phi)
        mdsInGPU.anchorZ[mdIndex2],                                      // inner T3 anchor hit 2 z (t3_2_z)
        alpaka::math::sqrt(acc, x2 * x2 + y2 * y2),                      // inner T3 anchor hit 2 r (t3_2_r)
        float(modulesInGPU.layers[lowerModuleIndex2] + 6 * is_endcap2),  // inner T3 anchor hit 2 layer (t3_2_layer)
        mdsInGPU.anchorEta[mdIndex3],                                    // inner T3 anchor hit 3 eta (t3_4_eta)
        mdsInGPU.anchorPhi[mdIndex3],                                    // inner T3 anchor hit 3 phi (t3_4_phi)
        mdsInGPU.anchorZ[mdIndex3],                                      // inner T3 anchor hit 3 z (t3_4_z)
        alpaka::math::sqrt(acc, x3 * x3 + y3 * y3),                      // inner T3 anchor hit 3 r (t3_4_r)
        float(modulesInGPU.layers[lowerModuleIndex3] + 6 * is_endcap3),  // inner T3 anchor hit 3 layer (t3_4_layer)
        SDL::temp_log10(acc, 2 * SDL::k2Rinv1GeVf * outerRadius),        // outer T3 pT (t3_pt)
        mdsInGPU.anchorEta[mdIndex3],                                    // outer T3 anchor hit 4 eta (t3_0_eta)
        mdsInGPU.anchorPhi[mdIndex3],                                    // outer T3 anchor hit 4 phi (t3_0_phi)
        mdsInGPU.anchorZ[mdIndex3],                                      // outer T3 anchor hit 3 eta (t3_0_z)
        alpaka::math::sqrt(acc, x3 * x3 + y3 * y3),                      // outer T3 anchor hit 3 r (t3_0_r)
        float(modulesInGPU.layers[lowerModuleIndex3] + 6 * is_endcap3),  // outer T3 anchor hit 3 layer (t3_0_layer)
        mdsInGPU.anchorEta[mdIndex4],                                    // outer T3 anchor hit 4 eta (t3_2_eta)
        mdsInGPU.anchorPhi[mdIndex4],                                    // outer T3 anchor hit 4 phi (t3_2_phi)
        mdsInGPU.anchorZ[mdIndex4],                                      // outer T3 anchor hit 4 z (t3_2_z)
        alpaka::math::sqrt(acc, x4 * x4 + y4 * y4),                      // outer T3 anchor hit 4 r (t3_2_r)
        float(modulesInGPU.layers[lowerModuleIndex4] + 6 * is_endcap4),  // outer T3 anchor hit 4 layer (t3_2_layer)
        mdsInGPU.anchorEta[mdIndex5],                                    // outer T3 anchor hit 5 eta (t3_4_eta)
        mdsInGPU.anchorPhi[mdIndex5],                                    // outer T3 anchor hit 5 phi (t3_4_phi)
        mdsInGPU.anchorZ[mdIndex5],                                      // outer T3 anchor hit 5 z (t3_4_z)
        alpaka::math::sqrt(acc, x5 * x5 + y5 * y5),                      // outer T3 anchor hit 5 r (t3_4_r)
        float(modulesInGPU.layers[lowerModuleIndex5] + 6 * is_endcap5),  // outer T3 anchor hit 5 layer (t3_4_layer)
        SDL::temp_log10(
            acc, (innerRadius + outerRadius) * SDL::magnetic_field * 1.602f / (2 * 100 * 5.39f)),  // T5 pT (t5_pt)
        mdsInGPU.anchorEta[md_idx_for_t5_eta_phi],                                                 // T5 eta (t5_eta)
        mdsInGPU.anchorPhi[md_idx_for_t5_eta_phi],                                                 // T5 phi (t5_phi)
        SDL::temp_log10(acc, innerRadius),   // T5 inner radius (t5_innerRadius)
        SDL::temp_log10(acc, bridgeRadius),  // T5 bridge radius (t5_bridgeRadius)
        SDL::temp_log10(acc, outerRadius)    // T5 outer radius (t5_outerRadius)
    };

    // (0): Linear(in_features=38, out_features=32, bias=True) => x = x*W_T + b
    float x_0[32];
    for (unsigned int col = 0; col < 32; ++col) {
      x_0[col] = 0;
      for (unsigned int inner = 0; inner < 38; ++inner) {
        x_0[col] += x[inner] * wgtT_0[inner][col];
      }
      x_0[col] += bias_0[col];
    }

    // (1): ReLU()
    float x_1[32];
    for (unsigned int col = 0; col < 32; ++col) {
      x_1[col] = (x_0[col] > 0.f) ? x_0[col] : 0.f;
    }

    // (2): Linear(in_features=32, out_features=32, bias=True) => x = x*W_T + b
    float x_2[32];
    for (unsigned int col = 0; col < 32; ++col) {
      x_2[col] = 0;
      for (unsigned int inner = 0; inner < 32; ++inner) {
        x_2[col] += x_1[inner] * wgtT_2[inner][col];
      }
      x_2[col] += bias_2[col];
    }

    // (3): ReLU()
    float x_3[32];
    for (unsigned int col = 0; col < 32; ++col) {
      x_3[col] = (x_2[col] > 0.f) ? x_2[col] : 0.f;
    }

    // (4): Linear(in_features=32, out_features=1, bias=True) => x = x*W_T + b
    float x_4[1];
    for (unsigned int col = 0; col < 1; ++col) {
      x_4[col] = 0;
      for (unsigned int inner = 0; inner < 32; ++inner) {
        x_4[col] += x_3[inner] * wgtT_4[inner][col];
      }
      x_4[col] += bias_4[col];
    }

    // (5): Sigmoid()
    float x_5[1];
    for (unsigned int col = 0; col < 1; ++col) {
      x_5[col] = alpaka::math::exp(acc, x_4[col]) / (alpaka::math::exp(acc, x_4[col]) + 1);
    }

    return x_5[0];
  }
}  // namespace T5DNN

#endif
