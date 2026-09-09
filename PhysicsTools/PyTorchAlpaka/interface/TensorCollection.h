#ifndef PhysicsTools_PyTorchAlpaka_interface_TensorCollection_h
#define PhysicsTools_PyTorchAlpaka_interface_TensorCollection_h

#include <map>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <alpaka/alpaka.hpp>
#include <ATen/core/ScalarType.h>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "PhysicsTools/PyTorch/interface/TorchInterface.h"
#include "PhysicsTools/PyTorchAlpaka/interface/TensorHandle.h"

namespace alpaka_cuda_async::torch {
  class AlpakaModel;
}

namespace alpaka_rocm_async::torch {
  class AlpakaModel;
}

namespace alpaka_serial_sync::torch {
  class AlpakaModel;
}

namespace cms::torch::alpakatools {

  template <typename T>
  bool check_location(int elements, const T* column) {
    return true;
  }

  template <typename T>
  bool check_location(int elements, const T* column, const T* other_column) {
    return (column + elements) == other_column;
  }

  template <typename T, typename... Others>
  bool check_location(int elements, const T* column, const T* other_column, Others... others) {
    return (column + elements) == other_column && check_location(elements, other_column, others...);
  }

  template <typename T, typename... Others>
  void assert_location(int elements, const T* column, const Others*... others) {
    bool ok = check_location(elements, column, others...);
    assert(ok && "Tensor columns are not contiguous in memory!");
  }

  template <typename T, typename... Others>
  concept SameTypes = (std::same_as<T, Others> && ...);

  template <typename TSoAParamsImpl, typename... Others>
  concept SameValueType = SameTypes<typename TSoAParamsImpl::ValueType, typename Others::ValueType...>;

  template <typename TSoAParamsImpl, typename... Others>
  concept SameScalarType = SameTypes<typename TSoAParamsImpl::ScalarType, typename Others::ScalarType...>;

  class TensorSlice {
  public:
    struct Bounds {
      uint32_t offset;
      uint32_t size;
    };

    TensorSlice() = default;

    TensorSlice(uint32_t batch_id, uint32_t batch_size) : batch_id_{batch_id}, batch_size_{batch_size}, full_{false} {}

    Bounds resolve(uint32_t total_size) const {
      if (full_)
        return {.offset = 0, .size = total_size};
      if (total_size == 0)
        return {.offset = 0, .size = 0};
      assert(batch_size_ != 0 && "Batch size should be greater than zero");
      assert(batch_id_ <= (total_size - 1) / batch_size_ && "Batch id is out of bounds!");
      const auto offset = batch_id_ * batch_size_;
      return {.offset = offset, .size = std::min(batch_size_, total_size - offset)};
    }

  private:
    uint32_t batch_id_ = 0;
    uint32_t batch_size_ = 0;
    bool full_ = true;
  };

  // Container for user defined memory blobs that will be converted to PyTorch tensors constructs directly from
  // provided recipies and contiguous memory blocks
  //
  // Provided memory blocks must be of the same type and to be contiguous e.g.:
  //
  // GENERATE_SOA_LAYOUT(ParticleLayout,
  //                     SOA_COLUMN(float, pt),
  //                     SOA_COLUMN(float, eta),
  //                     SOA_COLUMN(float, phi))
  //
  // can register the following:
  //
  // TensorCollection<Device> registry(batch_size);
  // registry.add<ParticleLayout>("features", batch_id, records.pt(), records.eta(), records.phi());
  //
  // In the above example, the add function automatically computes the offset for the batch and ensures the provided columns are contiguous in memory.
  // If the user wants to perform inference on the entire dataset without batching, he can simply register by passing just the total size:
  //
  // TensorCollection<Device> registry();
  // registry.add<ParticleLayout>("features", records.pt(), records.eta(), records.phi());
  //
  // If the user wants to use only pt() and phi() then below will not work as pt() and phi() are not contiguous:
  // TensorCollection<Device> registry(batch_size);
  // registry.add<ParticleLayout>("features", records.pt(), records.phi());
  //
  // potential solution would be to arrange layout dependent on model requirements
  // GENERATE_SOA_LAYOUT(ParticleLayout,
  //                     SOA_COLUMN(float, pt),
  //                     SOA_COLUMN(float, phi),  note features position was swapped to ensure continuity
  //                     SOA_COLUMN(float, eta))
  //

  template <typename TQueue>
    requires alpaka::isQueue<TQueue>
  class TensorCollection {
  public:
    friend class alpaka_cuda_async::torch::AlpakaModel;
    friend class alpaka_rocm_async::torch::AlpakaModel;
    friend class alpaka_serial_sync::torch::AlpakaModel;

    TensorCollection() = default;

    // SOA_EIGEN_COLUMN
    template <typename SoALayout, typename TSoAParamsImpl, typename... Others>
      requires(SameValueType<TSoAParamsImpl, Others...> && TSoAParamsImpl::columnType == cms::soa::SoAColumnType::eigen)
    void add(const std::string& name,
             TensorSlice slice,
             std::tuple<TSoAParamsImpl, cms::soa::size_type> column,
             std::tuple<Others, cms::soa::size_type>... others) {
      using DataType = typename TSoAParamsImpl::ScalarType;

      auto ptr = std::get<0>(column).data();
      const auto soa_size = std::get<1>(column);
      const auto [offset, effective_size] = slice.resolve(soa_size);

      int n_elems =
          cms::torch::alpakatools::detail::num_elements_per_column(soa_size, SoALayout::alignment, sizeof(DataType));
      assert_location(
          n_elems * TSoAParamsImpl::ValueType::RowsAtCompileTime * TSoAParamsImpl::ValueType::ColsAtCompileTime,
          ptr,
          std::get<0>(others).data()...);

      ptr += offset;

      std::vector<int> tensor_dims;
      if constexpr (TSoAParamsImpl::ValueType::ColsAtCompileTime > 1)
        tensor_dims = {1 + sizeof...(Others),
                       TSoAParamsImpl::ValueType::RowsAtCompileTime,
                       TSoAParamsImpl::ValueType::ColsAtCompileTime};
      else
        tensor_dims = {1 + sizeof...(Others), TSoAParamsImpl::ValueType::RowsAtCompileTime};

      emplace_tensor(name, SoALayout::alignment, ptr, effective_size, soa_size, tensor_dims);
    }

    // SOA_EIGEN_COLUMN with default batch size = default size
    template <typename SoALayout, typename TSoAParamsImpl, typename... Others>
      requires(SameValueType<TSoAParamsImpl, Others...> && TSoAParamsImpl::columnType == cms::soa::SoAColumnType::eigen)
    void add(const std::string& name,
             std::tuple<TSoAParamsImpl, cms::soa::size_type> column,
             std::tuple<Others, cms::soa::size_type>... others) {
      add<SoALayout, TSoAParamsImpl, Others...>(name, TensorSlice{}, column, others...);
    }

    // SOA_COLUMN
    template <typename SoALayout, typename TSoAParamsImpl, typename... Others>
      requires(SameScalarType<TSoAParamsImpl, Others...> &&
               TSoAParamsImpl::columnType == cms::soa::SoAColumnType::column)
    void add(const std::string& name,
             TensorSlice slice,
             std::tuple<TSoAParamsImpl, cms::soa::size_type> column,
             std::tuple<Others, cms::soa::size_type>... others) {
      using DataType = typename TSoAParamsImpl::ScalarType;

      auto ptr = std::get<0>(column).data();
      const auto soa_size = std::get<1>(column);
      const auto [offset, effective_size] = slice.resolve(soa_size);

      int n_elems =
          cms::torch::alpakatools::detail::num_elements_per_column(soa_size, SoALayout::alignment, sizeof(DataType));
      assert_location(n_elems, ptr, std::get<0>(others).data()...);

      ptr += offset;
      emplace_tensor(name, SoALayout::alignment, ptr, effective_size, soa_size, {1 + sizeof...(Others)});
    }

    // SOA_COLUMN with default batch size = total size
    template <typename SoALayout, typename TSoAParamsImpl, typename... Others>
      requires(SameScalarType<TSoAParamsImpl, Others...> &&
               TSoAParamsImpl::columnType == cms::soa::SoAColumnType::column)
    void add(const std::string& name,
             std::tuple<TSoAParamsImpl, cms::soa::size_type> column,
             std::tuple<Others, cms::soa::size_type>... others) {
      add<SoALayout, TSoAParamsImpl, Others...>(name, TensorSlice{}, column, others...);
    }

    // SOA_SCALAR
    template <typename SoALayout, cms::soa::SoAColumnType column_t, typename T>
      requires(std::is_arithmetic_v<T> && column_t == cms::soa::SoAColumnType::scalar)
    void add(const std::string& name,
             std::tuple<cms::soa::SoAParametersImpl<column_t, T>, cms::soa::size_type> column) {
      auto ptr = std::get<0>(column).data();
      const auto soa_size = std::get<1>(column);

      emplace_tensor(name, SoALayout::alignment, ptr, soa_size, soa_size, {1}, true);
    }

    // The order is defined by the order `add()` is called.
    // It can be changed by passing a vector of the block names afterwards.
    void change_order(std::vector<std::string> order) {
      assert(order.size() == order_.size() &&
             "TensorCollection::change_order: size mismatch, all blocks have to be mentioned.");
      order_ = std::move(order);
    }
    size_t size() const { return registry_.size(); }
    cms::torch::alpakatools::detail::ITensorHandle<TQueue>& operator[](const size_t index) const {
      return *registry_.at(order_[index]);
    }

  private:
    void copy(TQueue& queue) {
      for (const auto& name : order_)
        registry_.at(name)->copy(queue);
    }

    // propagate pointer (Tptr) and type to distinguish between T* and const T* and trigger internal copy.
    template <typename Tptr>
    void emplace_tensor(const std::string& name,
                        size_t alignment,
                        Tptr ptr,
                        int batch_size,
                        int total_size,
                        std::vector<int> dims = {1},
                        const bool is_scalar = false) {
      using T = std::remove_pointer_t<Tptr>;
      registry_.try_emplace(name,
                            std::make_unique<cms::torch::alpakatools::detail::TensorHandle<TQueue, T>>(
                                alignment, sizeof(T), ptr, batch_size, total_size, std::move(dims), is_scalar));
      order_.push_back(name);
    }

    std::vector<std::string> order_;
    std::unordered_map<std::string, std::unique_ptr<cms::torch::alpakatools::detail::ITensorHandle<TQueue>>> registry_;
  };

}  // namespace cms::torch::alpakatools

#endif  // PhysicsTools_PyTorchAlpaka_interface_TensorCollection_h
