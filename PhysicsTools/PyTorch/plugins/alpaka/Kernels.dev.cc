// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "PhysicsTools/PyTorch/plugins/alpaka/Kernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  class FillParticleCollectionKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, torchportable::ParticleCollection::View data, float value) const {
      for (auto tid : uniform_elements(acc, data.metadata().size())) {
        data.pt()[tid] = value;
        data.phi()[tid] = value;
        data.eta()[tid] = value;
      }
    }
  };

  /**
   * @brief Fill all values in a particle collection with a specified constant.
   * 
   * For debugging and unit testing.
   *
   * @param queue Alpaka execution queue.
   * @param data Particle collection to be modified.
   * @param value Constant value to fill the collection with.
   */
  void Kernels::FillParticleCollection(Queue &queue, torchportable::ParticleCollection &data, float value) {
    uint32_t threads_per_block = 512;
    uint32_t blocks_per_grid = divide_up_by(data.view().metadata().size(), threads_per_block);
    auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
    alpaka::exec<Acc1D>(queue, grid, FillParticleCollectionKernel{}, data.view(), value);
  }

  class AssertCombinatoricsKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, torchportable::ParticleCollection::View data, float value) const {
      for (auto tid : uniform_elements(acc, data.metadata().size())) {
        ALPAKA_ASSERT_ACC(data.pt()[tid] == value);
        ALPAKA_ASSERT_ACC(data.phi()[tid] == value);
        ALPAKA_ASSERT_ACC(data.eta()[tid] == value);
      }
    }
  };

  /**
   * @brief Assert that the particle collection obeys certain combinatoric relationships.
   *
   * Used in test scenarios to verify data layout or transformation logic.
   *
   * @param queue Alpaka execution queue.
   * @param data Particle collection to check.
   * @param value Reference value for validation.
   */
  void Kernels::AssertCombinatorics(Queue &queue, torchportable::ParticleCollection &data, float value) {
    uint32_t threads_per_block = 512;
    uint32_t blocks_per_grid = divide_up_by(data.view().metadata().size(), threads_per_block);
    auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
    alpaka::exec<Acc1D>(queue, grid, AssertCombinatoricsKernel{}, data.view(), value);
  }

  class AssertClassificationKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, torchportable::ClassificationCollection::View data) const {
      for (auto tid : uniform_elements(acc, data.metadata().size())) {
        ALPAKA_ASSERT_ACC(data.c1()[tid] == 0.5f);
        ALPAKA_ASSERT_ACC(data.c2()[tid] == 0.5f);
      }
    }
  };

  /**
   * @brief Validate classification model outputs.
   *
   * Checks whether the classification outputs match expected format or values.
   * Used for debugging and integration testing of model inference.
   *
   * @param queue Alpaka execution queue.
   * @param data Classification output collection.
   */
  void Kernels::AssertClassification(Queue &queue, torchportable::ClassificationCollection &data) {
    uint32_t threads_per_block = 512;
    uint32_t blocks_per_grid = divide_up_by(data.view().metadata().size(), threads_per_block);
    auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
    alpaka::exec<Acc1D>(queue, grid, AssertClassificationKernel{}, data.view());
  }

  class AssertRegressionKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, torchportable::RegressionCollection::View data) const {
      for (auto tid : uniform_elements(acc, data.metadata().size())) {
        ALPAKA_ASSERT_ACC(data.reco_pt()[tid] == 0.5f);
      }
    }
  };

  /**
   * @brief Validate regression model outputs.
   *
   * Similar to classification checks, this is used for asserting output correctness in regression tasks.
   *
   * @param queue Alpaka execution queue.
   * @param data Regression output collection.
   */
  void Kernels::AssertRegression(Queue &queue, torchportable::RegressionCollection &data) {
    uint32_t threads_per_block = 512;
    uint32_t blocks_per_grid = divide_up_by(data.view().metadata().size(), threads_per_block);
    auto grid = make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);
    alpaka::exec<Acc1D>(queue, grid, AssertRegressionKernel{}, data.view());
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
