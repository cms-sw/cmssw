#ifndef PhysicsTools_PyTorchAlpaka_interface_TensorCollection_h
#define PhysicsTools_PyTorchAlpaka_interface_TensorCollection_h

#include <map>
#include <string>
#include <tuple>
#include <type_traits>
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
  // TensorCollection<Device> registry(batch_size);
  // registry.add<ParticleLayout>("features", records.pt(), records.eta(), records.phi());
  //
  // but if want to use only pt() and phi() then below will not work as pt() and phi() are not contiguous:
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

    explicit TensorCollection(int batch_size) : batch_size_(batch_size) {}

    // SOA_EIGEN_COLUMN
    template <typename SoALayout, typename TSoAParamsImpl, typename... Others>
      requires(SameValueType<TSoAParamsImpl, Others...> && TSoAParamsImpl::columnType == cms::soa::SoAColumnType::eigen)
    void add(const std::string& name,
             int batch_size,
             std::tuple<TSoAParamsImpl, cms::soa::size_type> column,
             std::tuple<Others, cms::soa::size_type>... others) {
      using DataType = typename TSoAParamsImpl::ScalarType;
      auto [ptr, stride] = std::get<0>(column).tupleOrPointer();
      int n_elems =
          cms::torch::alpakatools::detail::num_elements_per_column(batch_size, SoALayout::alignment, sizeof(DataType));
      assert_location(
          n_elems * TSoAParamsImpl::ValueType::RowsAtCompileTime * TSoAParamsImpl::ValueType::ColsAtCompileTime,
          ptr,
          std::get<0>(std::get<0>(others).tupleOrPointer())...);

      std::vector<int> tensor_dims;
      if constexpr (TSoAParamsImpl::ValueType::ColsAtCompileTime > 1)
        tensor_dims = {1 + sizeof...(Others),
                       TSoAParamsImpl::ValueType::RowsAtCompileTime,
                       TSoAParamsImpl::ValueType::ColsAtCompileTime};
      else
        tensor_dims = {1 + sizeof...(Others), TSoAParamsImpl::ValueType::RowsAtCompileTime};

      emplace_tensor(name, SoALayout::alignment, ptr, batch_size, tensor_dims);
    }

    // SOA_EIGEN_COLUMN with default batch size
    template <typename SoALayout, typename TSoAParamsImpl, typename... Others>
      requires(SameValueType<TSoAParamsImpl, Others...> && TSoAParamsImpl::columnType == cms::soa::SoAColumnType::eigen)
    void add(const std::string& name,
             std::tuple<TSoAParamsImpl, cms::soa::size_type> column,
             std::tuple<Others, cms::soa::size_type>... others) {
      add<SoALayout, TSoAParamsImpl, Others...>(name, batch_size_, column, others...);
    }

    // SOA_COLUMN
    template <typename SoALayout, typename TSoAParamsImpl, typename... Others>
      requires(SameScalarType<TSoAParamsImpl, Others...> &&
               TSoAParamsImpl::columnType == cms::soa::SoAColumnType::column)
    void add(const std::string& name,
             int batch_size,
             std::tuple<TSoAParamsImpl, cms::soa::size_type> column,
             std::tuple<Others, cms::soa::size_type>... others) {
      using DataType = typename TSoAParamsImpl::ScalarType;
      int n_elems =
          cms::torch::alpakatools::detail::num_elements_per_column(batch_size, SoALayout::alignment, sizeof(DataType));
      assert_location(n_elems, std::get<0>(column).tupleOrPointer(), std::get<0>(others).tupleOrPointer()...);
      auto ptr = std::get<0>(column).tupleOrPointer();
      emplace_tensor(name, SoALayout::alignment, ptr, batch_size, {1 + sizeof...(Others)});
    }

    // SOA_COLUMN with default batch size
    template <typename SoALayout, typename TSoAParamsImpl, typename... Others>
      requires(SameScalarType<TSoAParamsImpl, Others...> &&
               TSoAParamsImpl::columnType == cms::soa::SoAColumnType::column)
    void add(const std::string& name,
             std::tuple<TSoAParamsImpl, cms::soa::size_type> column,
             std::tuple<Others, cms::soa::size_type>... others) {
      add<SoALayout, TSoAParamsImpl, Others...>(name, batch_size_, column, others...);
    }

    // SOA_SCALAR
    template <typename SoALayout, cms::soa::SoAColumnType column_t, typename T>
      requires(std::is_arithmetic_v<T> && column_t == cms::soa::SoAColumnType::scalar)
    void add(const std::string& name,
             int batch_size,
             std::tuple<cms::soa::SoAParametersImpl<column_t, T>, cms::soa::size_type> column) {
      auto ptr = std::get<0>(column).tupleOrPointer();
      emplace_tensor(name, SoALayout::alignment, ptr, batch_size, {1}, true);
    }

    // SOA_SCALAR with default batch size
    template <typename SoALayout, cms::soa::SoAColumnType column_t, typename T>
      requires(std::is_arithmetic_v<T> && column_t == cms::soa::SoAColumnType::scalar)
    void add(const std::string& name,
             std::tuple<cms::soa::SoAParametersImpl<column_t, T>, cms::soa::size_type> column) {
      add<SoALayout, column_t, T>(name, batch_size_, column);
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
    void copy(TQueue& queue, const cms::torch::alpakatools::detail::MemcpyKind kind) {
      for (const auto& name : order_)
        registry_.at(name)->copy(queue, kind);
      // explicit synchronize to ensure data is in place before inference
      // no need to explicitly synchronize D2D/H2D, rely on implicit synchronization mechanism in framework
      if (kind == cms::torch::alpakatools::detail::MemcpyKind::DeviceToHost)
        alpaka::wait(queue);
    }

    // propagate pointer (Tptr) and type to distinguish between T* and const T* and trigger internal copy.
    template <typename Tptr>
    void emplace_tensor(const std::string& name,
                        size_t alignment,
                        Tptr ptr,
                        int batch_size,
                        std::vector<int> dims = {1},
                        const bool is_scalar = false) {
      using T = std::remove_pointer_t<Tptr>;
      registry_.try_emplace(name,
                            std::make_unique<cms::torch::alpakatools::detail::TensorHandle<TQueue, T>>(
                                alignment, sizeof(T), ptr, batch_size, std::move(dims), is_scalar));
      order_.push_back(name);
    }

    int batch_size_;
    std::vector<std::string> order_;
    std::unordered_map<std::string, std::unique_ptr<cms::torch::alpakatools::detail::ITensorHandle<TQueue>>> registry_;
  };

}  // namespace cms::torch::alpakatools

#endif  // PhysicsTools_PyTorchAlpaka_interface_TensorCollection_h
