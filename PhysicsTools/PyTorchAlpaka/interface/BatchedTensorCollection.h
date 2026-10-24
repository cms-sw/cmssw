#ifndef PhysicsTools_PyTorchAlpaka_interface_BatchedTensorCollection_h
#define PhysicsTools_PyTorchAlpaka_interface_BatchedTensorCollection_h

#include <algorithm>
#include <functional>
#include <optional>

#include "PhysicsTools/PyTorchAlpaka/interface/Exception.h"
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

    // Count batches and check consistency within this collection.
    // Sliced registrations determine the count, including zero for an SoA with size 0.
    // Full registrations do not determine the count if any sliced registration exists.
    //
    // If only full SoAs are registered:
    //   - return 0 if all of them have size 0.
    //   - return 1 otherwise.

    uint32_t batchCount() const {
      auto count = 0u;
      bool found_batched_tensor = false;
      bool has_full_data = false;

      for (const auto& recipe : batch_recipes_) {
        if (!recipe.batch_size) {
          has_full_data |= recipe.total_size != 0;
          continue;
        }

        const auto n_batches = recipe.total_size == 0 ? 0u : 1u + (recipe.total_size - 1u) / *recipe.batch_size;

        if (found_batched_tensor) {
          if (count != n_batches)
            detail::throwException("BatchedTensorCollection", "inconsistent number of batches");
        } else {
          count = n_batches;
          found_batched_tensor = true;
        }
      }
      if (found_batched_tensor)
        return count;
      return has_full_data ? 1u : 0u;
    }

    // Register fresh handles for one batch.
    // The destination must not already contain any of the registered tensor names.
    void materializeBatch(uint32_t batch_id, TensorCollection<TQueue>& destination) {
      for (const auto& recipe : batch_recipes_) {
        recipe.materialize(destination, batch_id);
      }
    }

    // addBatched allows for registering an SoA with a fixed batch size
    // SOA_SCALAR is intentionally excluded and must be registered with add().
    template <typename SoALayout, typename First, typename... Others>
      requires(First::columnType != cms::soa::SoAColumnType::scalar)
    void addBatched(const std::string& name,
                    uint32_t batch_size,
                    std::tuple<First, cms::soa::size_type> column,
                    std::tuple<Others, cms::soa::size_type>... others) {
      checkName(name);

      if (batch_size == 0)
        detail::throwException("BatchedTensorCollection", "batch_size must be greater than 0");

      const auto soa_size = static_cast<uint32_t>(std::get<1>(column));

      auto materialize = [=](TensorCollection<TQueue>& destination, uint32_t batch_id) {
        destination.template add<SoALayout>(name, TensorSlice{batch_id, batch_size}, column, others...);
      };

      batch_recipes_.push_back(BatchRecipe{name, soa_size, batch_size, std::move(materialize)});
    }

    // Register the complete SoA for every batch, or an unsliced SOA_SCALAR.
    template <typename SoALayout, typename First, typename... Others>
    void add(const std::string& name,
             std::tuple<First, cms::soa::size_type> column,
             std::tuple<Others, cms::soa::size_type>... others) {
      checkName(name);
      const auto soa_size = static_cast<uint32_t>(std::get<1>(column));

      auto materialize = [=](TensorCollection<TQueue>& destination, uint32_t) {
        destination.template add<SoALayout>(name, column, others...);
      };

      batch_recipes_.push_back(BatchRecipe{name, soa_size, std::nullopt, std::move(materialize)});
    }

  private:
    void checkName(const std::string& name) const {
      const auto duplicate = std::any_of(
          batch_recipes_.begin(), batch_recipes_.end(), [&](const auto& recipe) { return recipe.name == name; });

      if (duplicate)
        detail::throwException("BatchedTensorCollection", "tensor name '" + name + "' is already registered.");
    }

    struct BatchRecipe {
      std::string name;
      uint32_t total_size;
      std::optional<uint32_t> batch_size;
      std::function<void(TensorCollection<TQueue>&, uint32_t)> materialize;
    };

    std::vector<BatchRecipe> batch_recipes_;
  };

}  // namespace cms::torch::alpakatools

#endif  // PhysicsTools_PyTorchAlpaka_interface_BatchedTensorCollection_h
