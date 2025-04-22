#ifndef PHYSICS_TOOLS__PYTORCH__PLUGINS__ALPAKA__KERNELS_H_
#define PHYSICS_TOOLS__PYTORCH__PLUGINS__ALPAKA__KERNELS_H_

#include <alpaka/alpaka.hpp>

#include "DataFormats/PyTorchTest/interface/alpaka/Collections.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  /**
   * @class Kernels
   * @brief Utility class containing helper functions to run simple Alpaka kernels for testing or validation.
   *
   * This class provides simple device-side functionality for modifying and verifying
   * collections of structured SoA data, such as particles, classification outputs, and regressions.
   */
  class Kernels {
  public:
    void FillParticleCollection(Queue &queue, torchportable::ParticleCollection &data, float value);
    void AssertCombinatorics(Queue &queue, torchportable::ParticleCollection &data, float value);
    void AssertClassification(Queue &queue, torchportable::ClassificationCollection &data);
    void AssertRegression(Queue &queue, torchportable::RegressionCollection &data);
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // PHYSICS_TOOLS__PYTORCH__PLUGINS__ALPAKA__KERNELS_H_
