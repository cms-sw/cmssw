#ifndef PhysicsTools_PyTorchAlpaka_interface_BatchedTensorCollection_h
#define PhysicsTools_PyTorchAlpaka_interface_BatchedTensorCollection_h

#include <functional>
#include <optional>

#include "PhysicsTools/PyTorchAlpaka/interface/TensorCollection.h"

namespace cms::torch::alpakatools {

  template <typename TQueue>
    requires alpaka::isQueue<TQueue>
  class BatchedTensorCollection {
  public:
    friend class alpaka_cuda_async::torch::AlpakaModel;
    friend class alpaka_rocm_async::torch::AlpakaModel;
    friend class alpaka_serial_sync::torch::AlpakaModel;

    BatchedTensorCollection() = default;

    // Count the number of batches and check their consistency
    std::optional<uint32_t> batchCount() const {
      std::optional<uint32_t> count;

      for (const auto& recipe : batch_recipes_) {
        if (!recipe.batch_size)
          continue;  // Full SoA: it does not determine the batch count.

        auto n_batches = recipe.total_size == 0 ? 0u : 1u + (recipe.total_size - 1u) / *recipe.batch_size;
        assert((!count || *count == n_batches) && "BatchedTensorCollection: inconsistent number of batches");
        count = n_batches;
      }

      return count;
    }

    void materializeBatch(uint32_t batch_id, TensorCollection<TQueue>& destination) {
      for (const auto& recipe : batch_recipes_) {
        if (recipe.batch_size) {
          recipe.materialize(destination, batch_id);
          continue;
        }
        // Materialize each full-SoA tensor only once.
        if (!full_tensors_.registry_.contains(recipe.name))
          recipe.materialize(full_tensors_, 0u);

        // Preserve the registration order while sharing its handle.
        destination.registry_.try_emplace(recipe.name, full_tensors_.registry_.at(recipe.name));
        destination.order_.push_back(recipe.name);
      }
    }

    // addBatched allows for registering an SoA with a fixed batch size
    template <typename SoALayout, typename First, typename... Others>
    void addBatched(const std::string& name,
                    uint32_t batch_size,
                    std::tuple<First, cms::soa::size_type> column,
                    std::tuple<Others, cms::soa::size_type>... others) {
      assert(batch_size != 0 && "BatchedTensorCollection: batch size must be positive");

      const auto soa_size = static_cast<uint32_t>(std::get<1>(column));
      auto materialize = [=](TensorCollection<TQueue>& destination, uint32_t batch_id) {
        destination.template add<SoALayout>(name, TensorSlice{batch_id, batch_size}, column, others...);
      };

      batch_recipes_.push_back(BatchRecipe{name, soa_size, batch_size, std::move(materialize)});
    }

    // if no batch size is passed it falls back to SoA size
    template <typename SoALayout, typename First, typename... Others>
    void addBatched(const std::string& name,
                    std::tuple<First, cms::soa::size_type> column,
                    std::tuple<Others, cms::soa::size_type>... others) {
      const auto soa_size = static_cast<uint32_t>(std::get<1>(column));
      auto materialize = [=](TensorCollection<TQueue>& destination, uint32_t) {
        destination.template add<SoALayout>(name, column, others...);
      };

      batch_recipes_.push_back(BatchRecipe{name, soa_size, std::nullopt, std::move(materialize)});
    }

  private:
    // struct to record the necessary metadata to register a batched tensor in the collection
    struct BatchRecipe {
      std::string name;
      uint32_t total_size;
      std::optional<uint32_t> batch_size;
      std::function<void(TensorCollection<TQueue>&, uint32_t)> materialize;
    };

    std::vector<BatchRecipe> batch_recipes_;
    std::vector<std::unique_ptr<TensorCollection<TQueue>>> materialized_batches_;
    TensorCollection<TQueue> full_tensors_;
  };
}  // namespace cms::torch::alpakatools

#endif  // PhysicsTools_PyTorchAlpaka_interface_BatchedTensorCollection_h
