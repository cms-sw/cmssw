#ifndef RecoTracker_LSTCore_src_alpaka_NeuralNetwork_h
#define RecoTracker_LSTCore_src_alpaka_NeuralNetwork_h

#include "FWCore/Utilities/interface/CMSUnrollLoop.h"
#include "HeterogeneousCore/AlpakaMath/interface/deltaPhi.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsSoA.h"

#include "NeuralNetworkWeights.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst::t5dnn {

  template <int FEATURES>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void relu_activation(float (&input)[FEATURES]) {
    CMS_UNROLL_LOOP
    for (unsigned int col = 0; col < FEATURES; ++col) {
      input[col] = (input[col] > 0.f) ? input[col] : 0.f;
    }
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float sigmoid_activation(TAcc const& acc, const float x) {
    return alpaka::math::exp(acc, x) / (alpaka::math::exp(acc, x) + 1.f);
  }

  template <int IN_FEATURES, int OUT_FEATURES>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void linear_layer(const float (&input)[IN_FEATURES],
                                                   float (&output)[OUT_FEATURES],
                                                   const float (&weights)[IN_FEATURES][OUT_FEATURES],
                                                   const float (&biases)[OUT_FEATURES]) {
    CMS_UNROLL_LOOP
    for (unsigned int i = 0; i < OUT_FEATURES; ++i) {
      output[i] = biases[i];
      CMS_UNROLL_LOOP
      for (int j = 0; j < IN_FEATURES; ++j) {
        output[i] += input[j] * weights[j][i];
      }
    }
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runInference(TAcc const& acc,
                                                   MiniDoubletsConst mds,
                                                   const unsigned int mdIndex1,
                                                   const unsigned int mdIndex2,
                                                   const unsigned int mdIndex3,
                                                   const unsigned int mdIndex4,
                                                   const unsigned int mdIndex5,
                                                   const float innerRadius,
                                                   const float outerRadius,
                                                   const float bridgeRadius) {
    // Constants
    constexpr unsigned int kinputFeatures = 23;
    constexpr unsigned int khiddenFeatures = 32;

    float eta1 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex1]);  // inner T3 anchor hit 1 eta (t3_0_eta)
    float eta2 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex2]);  // inner T3 anchor hit 2 eta (t3_2_eta)
    float eta3 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex3]);  // inner T3 anchor hit 3 eta (t3_4_eta)
    float eta4 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex4]);  // outer T3 anchor hit 4 eta (t3_2_eta)
    float eta5 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex5]);  // outer T3 anchor hit 5 eta (t3_4_eta)

    float phi1 = mds.anchorPhi()[mdIndex1];  // inner T3 anchor hit 1 phi (t3_0_phi)
    float phi2 = mds.anchorPhi()[mdIndex2];  // inner T3 anchor hit 2 phi (t3_2_phi)
    float phi3 = mds.anchorPhi()[mdIndex3];  // inner T3 anchor hit 3 phi (t3_4_phi)
    float phi4 = mds.anchorPhi()[mdIndex4];  // outer T3 anchor hit 4 phi (t3_2_phi)
    float phi5 = mds.anchorPhi()[mdIndex5];  // outer T3 anchor hit 5 phi (t3_4_phi)

    float z1 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex1]);  // inner T3 anchor hit 1 z (t3_0_z)
    float z2 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex2]);  // inner T3 anchor hit 2 z (t3_2_z)
    float z3 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex3]);  // inner T3 anchor hit 3 z (t3_4_z)
    float z4 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex4]);  // outer T3 anchor hit 4 z (t3_2_z)
    float z5 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex5]);  // outer T3 anchor hit 5 z (t3_4_z)

    float r1 = mds.anchorRt()[mdIndex1];  // inner T3 anchor hit 1 r (t3_0_r)
    float r2 = mds.anchorRt()[mdIndex2];  // inner T3 anchor hit 2 r (t3_2_r)
    float r3 = mds.anchorRt()[mdIndex3];  // inner T3 anchor hit 3 r (t3_4_r)
    float r4 = mds.anchorRt()[mdIndex4];  // outer T3 anchor hit 4 r (t3_2_r)
    float r5 = mds.anchorRt()[mdIndex5];  // outer T3 anchor hit 5 r (t3_4_r)

    // Build the input feature vector using pairwise differences after the first hit
    float x[kinputFeatures] = {
        eta1 / kEta_norm,                          // inner T3: First hit eta normalized
        alpaka::math::abs(acc, phi1) / kPhi_norm,  // inner T3: First hit phi normalized
        z1 / kZ_max,                               // inner T3: First hit z normalized
        r1 / kR_max,                               // inner T3: First hit r normalized

        eta2 - eta1,                                              // inner T3: Difference in eta between hit 2 and 1
        cms::alpakatools::deltaPhi(acc, phi2, phi1) / kPhi_norm,  // inner T3: Difference in phi between hit 2 and 1
        (z2 - z1) / kZ_max,  // inner T3: Difference in z between hit 2 and 1 normalized
        (r2 - r1) / kR_max,  // inner T3: Difference in r between hit 2 and 1 normalized

        eta3 - eta2,                                              // inner T3: Difference in eta between hit 3 and 2
        cms::alpakatools::deltaPhi(acc, phi3, phi2) / kPhi_norm,  // inner T3: Difference in phi between hit 3 and 2
        (z3 - z2) / kZ_max,  // inner T3: Difference in z between hit 3 and 2 normalized
        (r3 - r2) / kR_max,  // inner T3: Difference in r between hit 3 and 2 normalized

        eta4 - eta3,                                              // outer T3: Difference in eta between hit 4 and 3
        cms::alpakatools::deltaPhi(acc, phi4, phi3) / kPhi_norm,  // inner T3: Difference in phi between hit 4 and 3
        (z4 - z3) / kZ_max,  // outer T3: Difference in z between hit 4 and 3 normalized
        (r4 - r3) / kR_max,  // outer T3: Difference in r between hit 4 and 3 normalized

        eta5 - eta4,                                              // outer T3: Difference in eta between hit 5 and 4
        cms::alpakatools::deltaPhi(acc, phi5, phi4) / kPhi_norm,  // inner T3: Difference in phi between hit 5 and 4
        (z5 - z4) / kZ_max,  // outer T3: Difference in z between hit 5 and 4 normalized
        (r5 - r4) / kR_max,  // outer T3: Difference in r between hit 5 and 4 normalized

        alpaka::math::log10(acc, innerRadius),   // T5 inner radius (t5_innerRadius)
        alpaka::math::log10(acc, bridgeRadius),  // T5 bridge radius (t5_bridgeRadius)
        alpaka::math::log10(acc, outerRadius)    // T5 outer radius (t5_outerRadius)
    };

    float x_1[khiddenFeatures];  // Layer 1 output
    float x_2[khiddenFeatures];  // Layer 2 output
    float x_3[1];                // Layer 3 linear output

    // Layer 1: Linear + Relu
    linear_layer<kinputFeatures, khiddenFeatures>(x, x_1, wgtT_layer1, bias_layer1);
    relu_activation<khiddenFeatures>(x_1);

    // Layer 2: Linear + Relu
    linear_layer<khiddenFeatures, khiddenFeatures>(x_1, x_2, wgtT_layer2, bias_layer2);
    relu_activation<khiddenFeatures>(x_2);

    // Layer 3: Linear + Sigmoid
    linear_layer<khiddenFeatures, 1>(x_2, x_3, wgtT_output_layer, bias_output_layer);
    float x_5 = sigmoid_activation(acc, x_3[0]);

    // Get the bin index based on abs(eta) of first hit and t5_pt
    float t5_pt = innerRadius * lst::k2Rinv1GeVf * 2;

    uint8_t pt_index = (t5_pt > 5);
    uint8_t bin_index = (eta1 > 2.5f) ? (kEtaBins - 1) : static_cast<unsigned int>(eta1 / 0.25f);

    // Compare x_5 to the cut value for the relevant bin
    return x_5 > kWp[pt_index][bin_index];
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst::t5dnn

#endif
