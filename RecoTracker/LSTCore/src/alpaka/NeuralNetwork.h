#ifndef RecoTracker_LSTCore_src_alpaka_NeuralNetwork_h
#define RecoTracker_LSTCore_src_alpaka_NeuralNetwork_h

#include "FWCore/Utilities/interface/CMSUnrollLoop.h"
#include "HeterogeneousCore/AlpakaMath/interface/deltaPhi.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsSoA.h"

#include "T5NeuralNetworkWeights.h"
#include "T3NeuralNetworkWeights.h"
#include "pT3NeuralNetworkWeights.h"
#include "T5EmbedNetworkWeights.h"
#include "pLSEmbedNetworkWeights.h"
#include "T4NeuralNetworkWeights.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  template <int FEATURES, alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void softmax_activation(TAcc const& acc, float (&input)[FEATURES]) {
    float sum = 0.f;
    // Compute exp and sum
    CMS_UNROLL_LOOP
    for (unsigned int i = 0; i < FEATURES; ++i) {
      input[i] = alpaka::math::exp(acc, input[i]);
      sum += input[i];
    }

    // Normalize
    CMS_UNROLL_LOOP
    for (unsigned int i = 0; i < FEATURES; ++i) {
      input[i] /= sum;
    }
  }

  template <int FEATURES>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void relu_activation(float (&input)[FEATURES]) {
    CMS_UNROLL_LOOP
    for (unsigned int col = 0; col < FEATURES; ++col) {
      input[col] = (input[col] > 0.f) ? input[col] : 0.f;
    }
  }

  template <alpaka::concepts::Acc TAcc>
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

  namespace t3dnn {
    template <alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runInference(TAcc const& acc,
                                                     MiniDoubletsConst mds,
                                                     const unsigned int mdIndex1,
                                                     const unsigned int mdIndex2,
                                                     const unsigned int mdIndex3,
                                                     const float radius,
                                                     const float betaIn,
                                                     float (&output)[dnn::t3dnn::kOutputFeatures]) {
      // Constants for T3 DNN
      constexpr unsigned int kInputFeatures = 14;
      constexpr unsigned int kHiddenFeatures = 32;

      // Extract hit information
      float eta1 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex1]);  // inner T3 anchor hit 1 eta
      float eta2 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex2]);  // inner T3 anchor hit 2 eta
      float eta3 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex3]);  // inner T3 anchor hit 3 eta

      float phi1 = mds.anchorPhi()[mdIndex1];  // inner T3 anchor hit 1 phi
      float phi2 = mds.anchorPhi()[mdIndex2];  // inner T3 anchor hit 2 phi
      float phi3 = mds.anchorPhi()[mdIndex3];  // inner T3 anchor hit 3 phi

      float z1 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex1]);  // inner T3 anchor hit 1 z
      float z2 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex2]);  // inner T3 anchor hit 2 z
      float z3 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex3]);  // inner T3 anchor hit 3 z

      float r1 = mds.anchorRt()[mdIndex1];  // inner T3 anchor hit 1 r
      float r2 = mds.anchorRt()[mdIndex2];  // inner T3 anchor hit 2 r
      float r3 = mds.anchorRt()[mdIndex3];  // inner T3 anchor hit 3 r

      // Build input feature vector matching training order
      float x[kInputFeatures] = {
          eta1 / dnn::t3dnn::kEta_norm,                   // First hit eta normalized
          alpaka::math::abs(acc, phi1) / dnn::kPhi_norm,  // First hit phi normalized
          z1 / dnn::t3dnn::kZ_max,                        // First hit z normalized
          r1 / dnn::t3dnn::kR_max,                        // First hit r normalized

          eta2 - eta1,                                                   // Difference in eta between hit 2 and 1
          cms::alpakatools::deltaPhi(acc, phi2, phi1) / dnn::kPhi_norm,  // Difference in phi between hit 2 and 1
          (z2 - z1) / dnn::t3dnn::kZ_max,  // Difference in z between hit 2 and 1 normalized
          (r2 - r1) / dnn::t3dnn::kR_max,  // Difference in r between hit 2 and 1 normalized

          eta3 - eta2,                                                   // Difference in eta between hit 3 and 2
          cms::alpakatools::deltaPhi(acc, phi3, phi2) / dnn::kPhi_norm,  // Difference in phi between hit 3 and 2
          (z3 - z2) / dnn::t3dnn::kZ_max,  // Difference in z between hit 3 and 2 normalized
          (r3 - r2) / dnn::t3dnn::kR_max,  // Difference in r between hit 3 and 2 normalized

          alpaka::math::log10(acc, radius),  // T3's circle radius
          betaIn                             // Beta angle of inner segment
      };

      float x_1[kHiddenFeatures];  // Layer 1 output
      float x_2[kHiddenFeatures];  // Layer 2 output

      // Layer 1: Linear + Relu
      linear_layer<kInputFeatures, kHiddenFeatures>(x, x_1, dnn::t3dnn::wgtT_layer1, dnn::t3dnn::bias_layer1);
      relu_activation<kHiddenFeatures>(x_1);

      // Layer 2: Linear + Relu
      linear_layer<kHiddenFeatures, kHiddenFeatures>(x_1, x_2, dnn::t3dnn::wgtT_layer2, dnn::t3dnn::bias_layer2);
      relu_activation<kHiddenFeatures>(x_2);

      // Layer 3: Linear + Softmax
      linear_layer<kHiddenFeatures, dnn::t3dnn::kOutputFeatures>(
          x_2, output, dnn::t3dnn::wgtT_output_layer, dnn::t3dnn::bias_output_layer);
      softmax_activation<dnn::t3dnn::kOutputFeatures>(acc, output);

      // Get pt and eta bin indices
      float t3_pt = radius * lst::k2Rinv1GeVf * 2;
      uint8_t pt_index = (t3_pt > 5);
      uint8_t bin_index = (eta1 > 2.5f) ? (dnn::kEtaBins - 1) : static_cast<unsigned int>(eta1 / dnn::kEtaSize);

      return output[1] > dnn::t3dnn::kWp_prompt[pt_index][bin_index] ||
             output[2] > dnn::t3dnn::kWp_displaced[pt_index][bin_index];
    }
  }  // namespace t3dnn

  namespace pt3dnn {

    template <typename WP, alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runInference(TAcc const& acc,
                                                     const float rPhiChiSquared,
                                                     const float tripletRadius,
                                                     const float pixelRadius,
                                                     const float pixRadiusError,
                                                     const float rzChiSquared,
                                                     const float pixelEta,
                                                     const float pixelPt,
                                                     const int moduleType3) {
      constexpr unsigned int kInputFeatures = 7;
      constexpr unsigned int kHiddenFeatures = 32;
      constexpr unsigned int kOutputFeatures = 1;

      float x[kInputFeatures] = {
          alpaka::math::log10(acc, rPhiChiSquared),
          alpaka::math::log10(acc, tripletRadius),
          alpaka::math::log10(acc, pixelRadius),
          alpaka::math::log10(acc, pixRadiusError),
          alpaka::math::log10(acc, rzChiSquared),
          alpaka::math::abs(acc, pixelEta) / dnn::pt3dnn::kEta_norm,
          static_cast<float>(moduleType3),
      };

      float x1[kHiddenFeatures];
      float x2[kHiddenFeatures];
      float x3[kOutputFeatures];

      linear_layer<kInputFeatures, kHiddenFeatures>(x, x1, dnn::pt3dnn::wgtT_layer1, dnn::pt3dnn::bias_layer1);
      relu_activation<kHiddenFeatures>(x1);

      linear_layer<kHiddenFeatures, kHiddenFeatures>(x1, x2, dnn::pt3dnn::wgtT_layer2, dnn::pt3dnn::bias_layer2);
      relu_activation<kHiddenFeatures>(x2);

      linear_layer<kHiddenFeatures, kOutputFeatures>(
          x2, x3, dnn::pt3dnn::wgtT_output_layer, dnn::pt3dnn::bias_output_layer);
      float output = sigmoid_activation(acc, x3[0]);

      uint8_t bin_index = (alpaka::math::abs(acc, pixelEta) > 2.5f)
                              ? (dnn::kEtaBins - 1)
                              : static_cast<unsigned int>(alpaka::math::abs(acc, pixelEta) / dnn::kEtaSize);

      if (pixelPt > 5.0f)
        return output > WP::wpHigh();

      return output > WP::wp(bin_index);
    }

  }  // namespace pt3dnn

  namespace t5dnn {
    template <alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runInference(TAcc const& acc,
                                                     MiniDoubletsConst mds,
                                                     const unsigned int mdIndex1,
                                                     const unsigned int mdIndex2,
                                                     const unsigned int mdIndex3,
                                                     const unsigned int mdIndex4,
                                                     const unsigned int mdIndex5,
                                                     const float innerRadius,
                                                     const float outerRadius,
                                                     const float bridgeRadius,
                                                     float& dnnScore) {
      // Constants
      constexpr unsigned int kInputFeatures = 23;
      constexpr unsigned int kHiddenFeatures = 32;

      float eta1 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex1]);  // inner T3 anchor hit 1 eta
      float eta2 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex2]);  // inner T3 anchor hit 2 eta
      float eta3 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex3]);  // inner T3 anchor hit 3 eta
      float eta4 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex4]);  // outer T3 anchor hit 4 eta
      float eta5 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex5]);  // outer T3 anchor hit 5 eta

      float phi1 = mds.anchorPhi()[mdIndex1];  // inner T3 anchor hit 1 phi
      float phi2 = mds.anchorPhi()[mdIndex2];  // inner T3 anchor hit 2 phi
      float phi3 = mds.anchorPhi()[mdIndex3];  // inner T3 anchor hit 3 phi
      float phi4 = mds.anchorPhi()[mdIndex4];  // outer T3 anchor hit 4 phi
      float phi5 = mds.anchorPhi()[mdIndex5];  // outer T3 anchor hit 5 phi

      float z1 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex1]);  // inner T3 anchor hit 1 z
      float z2 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex2]);  // inner T3 anchor hit 2 z
      float z3 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex3]);  // inner T3 anchor hit 3 z
      float z4 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex4]);  // outer T3 anchor hit 4 z
      float z5 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex5]);  // outer T3 anchor hit 5 z

      float r1 = mds.anchorRt()[mdIndex1];  // inner T3 anchor hit 1 r
      float r2 = mds.anchorRt()[mdIndex2];  // inner T3 anchor hit 2 r
      float r3 = mds.anchorRt()[mdIndex3];  // inner T3 anchor hit 3 r
      float r4 = mds.anchorRt()[mdIndex4];  // outer T3 anchor hit 4 r
      float r5 = mds.anchorRt()[mdIndex5];  // outer T3 anchor hit 5 r

      // Build the input feature vector using pairwise differences after the first hit
      float x[kInputFeatures] = {
          eta1 / dnn::t5dnn::kEta_norm,                   // inner T3: First hit eta normalized
          alpaka::math::abs(acc, phi1) / dnn::kPhi_norm,  // inner T3: First hit phi normalized
          z1 / dnn::t5dnn::kZ_max,                        // inner T3: First hit z normalized
          r1 / dnn::t5dnn::kR_max,                        // inner T3: First hit r normalized

          eta2 - eta1,  // inner T3: Difference in eta between hit 2 and 1
          cms::alpakatools::deltaPhi(acc, phi2, phi1) /
              dnn::kPhi_norm,              // inner T3: Difference in phi between hit 2 and 1
          (z2 - z1) / dnn::t5dnn::kZ_max,  // inner T3: Difference in z between hit 2 and 1 normalized
          (r2 - r1) / dnn::t5dnn::kR_max,  // inner T3: Difference in r between hit 2 and 1 normalized

          eta3 - eta2,  // inner T3: Difference in eta between hit 3 and 2
          cms::alpakatools::deltaPhi(acc, phi3, phi2) /
              dnn::kPhi_norm,              // inner T3: Difference in phi between hit 3 and 2
          (z3 - z2) / dnn::t5dnn::kZ_max,  // inner T3: Difference in z between hit 3 and 2 normalized
          (r3 - r2) / dnn::t5dnn::kR_max,  // inner T3: Difference in r between hit 3 and 2 normalized

          eta4 - eta3,  // outer T3: Difference in eta between hit 4 and 3
          cms::alpakatools::deltaPhi(acc, phi4, phi3) /
              dnn::kPhi_norm,              // outer T3: Difference in phi between hit 4 and 3
          (z4 - z3) / dnn::t5dnn::kZ_max,  // outer T3: Difference in z between hit 4 and 3 normalized
          (r4 - r3) / dnn::t5dnn::kR_max,  // outer T3: Difference in r between hit 4 and 3 normalized

          eta5 - eta4,  // outer T3: Difference in eta between hit 5 and 4
          cms::alpakatools::deltaPhi(acc, phi5, phi4) /
              dnn::kPhi_norm,              // outer T3: Difference in phi between hit 5 and 4
          (z5 - z4) / dnn::t5dnn::kZ_max,  // outer T3: Difference in z between hit 5 and 4 normalized
          (r5 - r4) / dnn::t5dnn::kR_max,  // outer T3: Difference in r between hit 5 and 4 normalized

          alpaka::math::log10(acc, innerRadius),   // T5 inner radius
          alpaka::math::log10(acc, bridgeRadius),  // T5 bridge radius
          alpaka::math::log10(acc, outerRadius)    // T5 outer radius
      };

      float x_1[kHiddenFeatures];  // Layer 1 output
      float x_2[kHiddenFeatures];  // Layer 2 output
      float x_3[1];                // Layer 3 linear output

      // Layer 1: Linear + Relu
      linear_layer<kInputFeatures, kHiddenFeatures>(x, x_1, dnn::t5dnn::wgtT_layer1, dnn::t5dnn::bias_layer1);
      relu_activation<kHiddenFeatures>(x_1);

      // Layer 2: Linear + Relu
      linear_layer<kHiddenFeatures, kHiddenFeatures>(x_1, x_2, dnn::t5dnn::wgtT_layer2, dnn::t5dnn::bias_layer2);
      relu_activation<kHiddenFeatures>(x_2);

      // Layer 3: Linear + Sigmoid
      linear_layer<kHiddenFeatures, 1>(x_2, x_3, dnn::t5dnn::wgtT_output_layer, dnn::t5dnn::bias_output_layer);
      dnnScore = sigmoid_activation(acc, x_3[0]);

      // Get the bin index based on abs(eta) of first hit and t5_pt
      float t5_pt = innerRadius * lst::k2Rinv1GeVf * 2;

      uint8_t pt_index = (t5_pt > 5.0f);
      uint8_t bin_index = (eta1 > 2.5f) ? (dnn::kEtaBins - 1) : static_cast<unsigned int>(eta1 / dnn::kEtaSize);

      // Compare output to the cut value for the relevant bin
      return dnnScore > dnn::t5dnn::kWp[pt_index][bin_index];
    }
  }  // namespace t5dnn

  namespace t5embdnn {
    template <alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void runEmbed(TAcc const& acc,
                                                 MiniDoubletsConst mds,
                                                 unsigned int mdIndex1,
                                                 unsigned int mdIndex2,
                                                 unsigned int mdIndex3,
                                                 unsigned int mdIndex4,
                                                 unsigned int mdIndex5,
                                                 float innerRadius,
                                                 float outerRadius,
                                                 float bridgeRadius,
                                                 float fakeScore1,
                                                 float promptScore1,
                                                 float dispScore1,
                                                 float fakeScore2,
                                                 float promptScore2,
                                                 float dispScore2,
                                                 float (&embedding)[Params_T5::kEmbed]) {
      constexpr unsigned int kInputFeatures = 30;
      constexpr unsigned int kHiddenFeatures = 32;

      float eta1 = mds.anchorEta()[mdIndex1];
      float eta2 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex2]);
      float eta3 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex3]);
      float eta4 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex4]);
      float eta5 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex5]);

      float phi1 = mds.anchorPhi()[mdIndex1];
      float phi2 = mds.anchorPhi()[mdIndex2];
      float phi3 = mds.anchorPhi()[mdIndex3];
      float phi4 = mds.anchorPhi()[mdIndex4];
      float phi5 = mds.anchorPhi()[mdIndex5];

      float z1 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex1]);
      float z2 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex2]);
      float z3 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex3]);
      float z4 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex4]);
      float z5 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex5]);

      float r1 = mds.anchorRt()[mdIndex1];
      float r2 = mds.anchorRt()[mdIndex2];
      float r3 = mds.anchorRt()[mdIndex3];
      float r4 = mds.anchorRt()[mdIndex4];
      float r5 = mds.anchorRt()[mdIndex5];

      float x[kInputFeatures] = {eta1 / dnn::t5dnn::kEta_norm,
                                 alpaka::math::cos(acc, phi1),
                                 alpaka::math::sin(acc, phi1),
                                 z1 / dnn::t5dnn::kZ_max,
                                 r1 / dnn::t5dnn::kR_max,

                                 eta2 - alpaka::math::abs(acc, eta1),
                                 cms::alpakatools::deltaPhi(acc, phi2, phi1),
                                 (z2 - z1) / dnn::t5dnn::kZ_max,
                                 (r2 - r1) / dnn::t5dnn::kR_max,

                                 eta3 - eta2,
                                 cms::alpakatools::deltaPhi(acc, phi3, phi2),
                                 (z3 - z2) / dnn::t5dnn::kZ_max,
                                 (r3 - r2) / dnn::t5dnn::kR_max,

                                 eta4 - eta3,
                                 cms::alpakatools::deltaPhi(acc, phi4, phi3),
                                 (z4 - z3) / dnn::t5dnn::kZ_max,
                                 (r4 - r3) / dnn::t5dnn::kR_max,

                                 eta5 - eta4,
                                 cms::alpakatools::deltaPhi(acc, phi5, phi4),
                                 (z5 - z4) / dnn::t5dnn::kZ_max,
                                 (r5 - r4) / dnn::t5dnn::kR_max,

                                 1.0f / innerRadius,
                                 1.0f / bridgeRadius,
                                 1.0f / outerRadius,

                                 fakeScore1,
                                 promptScore1,
                                 dispScore1,
                                 (fakeScore2 - fakeScore1),
                                 (promptScore2 - promptScore1),
                                 (dispScore2 - dispScore1)};

      float h1[kHiddenFeatures];
      float h2[kHiddenFeatures];

      linear_layer<kInputFeatures, kHiddenFeatures>(x, h1, dnn::t5embdnn::wgtT_fc1, dnn::t5embdnn::bias_fc1);
      relu_activation<kHiddenFeatures>(h1);

      linear_layer<kHiddenFeatures, kHiddenFeatures>(h1, h2, dnn::t5embdnn::wgtT_fc2, dnn::t5embdnn::bias_fc2);
      relu_activation<kHiddenFeatures>(h2);

      linear_layer<kHiddenFeatures, Params_T5::kEmbed>(h2, embedding, dnn::t5embdnn::wgtT_fc3, dnn::t5embdnn::bias_fc3);
    }

  }  // namespace t5embdnn

  namespace plsembdnn {
    template <alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void runEmbed(TAcc const& acc,
                                                 const float eta,
                                                 const float etaErr,
                                                 const float phi,
                                                 const float circleCenterX,
                                                 const float circleCenterY,
                                                 const float circleRadius,
                                                 const float ptIn,
                                                 const float ptErr,
                                                 const bool isQuad,
                                                 float (&embedding)[Params_pLS::kEmbed]) {
      constexpr unsigned int kInputFeatures = 10;
      constexpr unsigned int kHiddenFeatures = 32;

      float x[kInputFeatures] = {eta / dnn::plsembdnn::kEta_norm,
                                 etaErr / dnn::plsembdnn::kEtaErr_norm,
                                 alpaka::math::cos(acc, phi),
                                 alpaka::math::sin(acc, phi),
                                 1.0f / ptIn,
                                 alpaka::math::log10(acc, ptErr),
                                 isQuad ? 1.0f : 0.0f,
                                 alpaka::math::log10(acc, alpaka::math::abs(acc, circleCenterX)),
                                 alpaka::math::log10(acc, alpaka::math::abs(acc, circleCenterY)),
                                 alpaka::math::log10(acc, circleRadius)};

      float h1[kHiddenFeatures];
      float h2[kHiddenFeatures];

      linear_layer<kInputFeatures, kHiddenFeatures>(x, h1, dnn::plsembdnn::wgtT_fc1, dnn::plsembdnn::bias_fc1);
      relu_activation<kHiddenFeatures>(h1);

      linear_layer<kHiddenFeatures, kHiddenFeatures>(h1, h2, dnn::plsembdnn::wgtT_fc2, dnn::plsembdnn::bias_fc2);
      relu_activation<kHiddenFeatures>(h2);

      linear_layer<kHiddenFeatures, Params_pLS::kEmbed>(
          h2, embedding, dnn::plsembdnn::wgtT_fc3, dnn::plsembdnn::bias_fc3);
    }

  }  // namespace plsembdnn

  namespace t4dnn {
    template <alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runInference(TAcc const& acc,
                                                     MiniDoubletsConst mds,
                                                     ModulesConst modules,
                                                     const unsigned int mdIndex1,
                                                     const unsigned int mdIndex2,
                                                     const unsigned int mdIndex3,
                                                     const unsigned int mdIndex4,
                                                     uint16_t lowerModuleIndex1,
                                                     uint16_t lowerModuleIndex2,
                                                     uint16_t lowerModuleIndex3,
                                                     uint16_t lowerModuleIndex4,
                                                     const float innerRadius,
                                                     const float outerRadius,
                                                     float& promptScore,
                                                     float& displacedScore,
                                                     float& fakeScore,
                                                     const float regressionRadius,
                                                     const float nonAnchorRegressionRadius,
                                                     float fakeScore1,
                                                     float promptScore1,
                                                     float displacedScore1,
                                                     float fakeScore2,
                                                     float promptScore2,
                                                     float displacedScore2) {
      // Constants
      constexpr unsigned int kinputFeatures = 30;
      constexpr unsigned int khiddenFeatures = 32;
      constexpr unsigned int koutputFeatures = 3;

      float eta1 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex1]);  // inner T3 anchor hit 1 eta (t3_0_eta)
      float eta2 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex2]);  // inner T3 anchor hit 2 eta (t3_2_eta)
      float eta3 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex3]);  // inner T3 anchor hit 3 eta (t3_4_eta)
      float eta4 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex4]);  // outer T3 anchor hit 4 eta (t3_2_eta)

      float phi1 = mds.anchorPhi()[mdIndex1];  // inner T3 anchor hit 1 phi
      float phi2 = mds.anchorPhi()[mdIndex2];  // inner T3 anchor hit 2 phi
      float phi3 = mds.anchorPhi()[mdIndex3];  // inner T3 anchor hit 3 phi
      float phi4 = mds.anchorPhi()[mdIndex4];  // outer T3 anchor hit 4 phi

      float z1 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex1]);  // inner T3 anchor hit 1 z (t3_0_z)
      float z2 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex2]);  // inner T3 anchor hit 2 z (t3_2_z)
      float z3 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex3]);  // inner T3 anchor hit 3 z (t3_4_z)
      float z4 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex4]);  // outer T3 anchor hit 4 z (t3_2_z)

      float r1 = mds.anchorRt()[mdIndex1];  // inner T3 anchor hit 1 r (t3_0_r)
      float r2 = mds.anchorRt()[mdIndex2];  // inner T3 anchor hit 2 r (t3_2_r)
      float r3 = mds.anchorRt()[mdIndex3];  // inner T3 anchor hit 3 r (t3_4_r)
      float r4 = mds.anchorRt()[mdIndex4];  // outer T3 anchor hit 4 r (t3_2_r)

      // Build the input feature vector using pairwise differences after the first hit
      float x[kinputFeatures] = {
          eta1 / dnn::t4dnn::kEta_norm,                   // inner T3: First hit eta normalized
          alpaka::math::abs(acc, phi1) / dnn::kPhi_norm,  // inner T3: First hit phi normalized
          z1 / dnn::t4dnn::kZ_max,                        // inner T3: First hit z normalized
          r1 / dnn::t4dnn::kR_max,                        // inner T3: First hit r normalized

          eta2 - eta1,  // inner T3: Difference in eta between hit 2 and 1
          cms::alpakatools::deltaPhi(acc, phi2, phi1) /
              dnn::kPhi_norm,              // inner T3: Difference in phi between hit 2 and 1
          (z2 - z1) / dnn::t4dnn::kZ_max,  // inner T3: Difference in z between hit 2 and 1 normalized
          (r2 - r1) / dnn::t4dnn::kR_max,  // inner T3: Difference in r between hit 2 and 1 normalized

          eta3 - eta2,  // inner T3: Difference in eta between hit 3 and 2
          cms::alpakatools::deltaPhi(acc, phi3, phi2) /
              dnn::kPhi_norm,              // inner T3: Difference in phi between hit 3 and 2
          (z3 - z2) / dnn::t4dnn::kZ_max,  // inner T3: Difference in z between hit 3 and 2 normalized
          (r3 - r2) / dnn::t4dnn::kR_max,  // inner T3: Difference in r between hit 3 and 2 normalized

          eta4 - eta3,  // outer T3: Difference in eta between hit 4 and 3
          cms::alpakatools::deltaPhi(acc, phi4, phi3) /
              dnn::kPhi_norm,              // inner T3: Difference in phi between hit 4 and 3
          (z4 - z3) / dnn::t4dnn::kZ_max,  // outer T3: Difference in z between hit 4 and 3 normalized
          (r4 - r3) / dnn::t4dnn::kR_max,  // outer T3: Difference in r between hit 4 and 3 normalized

          1.0f / innerRadius,  // T4 inner radius (t4_innerRadius)
          1.0f / outerRadius,  // T4 outer radius (t4_outerRadius)
          innerRadius / outerRadius,
          1.0f / regressionRadius,
          1.0f / nonAnchorRegressionRadius,
          fakeScore1,
          promptScore1,
          displacedScore1,
          fakeScore2,
          promptScore2,
          displacedScore2,
          (fakeScore2 - fakeScore1),
          (promptScore2 - promptScore1),
          (displacedScore2 - displacedScore1),
      };

      float x_1[khiddenFeatures];  // Layer 1 output
      float x_2[khiddenFeatures];  // Layer 2 output
      float x_3[koutputFeatures];  // Layer 3 output

      // Layer 1: Linear + Relu
      linear_layer<kinputFeatures, khiddenFeatures>(x, x_1, dnn::t4dnn::wgtT_layer1, dnn::t4dnn::bias_layer1);
      relu_activation<khiddenFeatures>(x_1);

      // Layer 2: Linear + Relu
      linear_layer<khiddenFeatures, khiddenFeatures>(x_1, x_2, dnn::t4dnn::wgtT_layer2, dnn::t4dnn::bias_layer2);
      relu_activation<khiddenFeatures>(x_2);

      // Layer 3: Linear + Softmax
      linear_layer<khiddenFeatures, koutputFeatures>(
          x_2, x_3, dnn::t4dnn::wgtT_output_layer, dnn::t4dnn::bias_output_layer);
      softmax_activation<koutputFeatures>(acc, x_3);

      // Get the bin index based on abs(eta) of first hit and t4_pt
      float t4_pt = (innerRadius + outerRadius) * lst::k2Rinv1GeVf;  //t4 pt is average

      uint8_t pt_index = (t4_pt > 5.f);
      uint8_t bin_index = (eta1 > 2.5f) ? (dnn::t4dnn::kEtaBins - 1) : static_cast<unsigned int>(eta1 / 0.1f);

      promptScore = x_3[1];
      displacedScore = x_3[2];
      fakeScore = x_3[0];

      return (x_3[2] > dnn::t4dnn::kWp_displaced[pt_index][bin_index]) &&
             (x_3[0] < dnn::t4dnn::kWp_fake[pt_index][bin_index]);
    }

  }  //namespace t4dnn

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif
