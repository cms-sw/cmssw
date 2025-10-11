#ifndef PhysicsTools_PyTorchAlpaka_interface_TensorRegistry_h
#define PhysicsTools_PyTorchAlpaka_interface_TensorRegistry_h

#include <map>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <ATen/core/ScalarType.h>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "PhysicsTools/PyTorch/interface/TorchInterface.h"
#include "PhysicsTools/PyTorchAlpaka/interface/Policy.h"
#include "PhysicsTools/PyTorchAlpaka/interface/TensorHandle.h"

namespace cms::torch::alpakatools {

  using namespace cms::soa;
  using namespace cms::alpakatools;

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

  template <typename T>
  ::torch::ScalarType get_type() {
    return ::torch::CppTypeToScalarType<T>();
  }

  class TensorRegistry {
  public:
    explicit TensorRegistry(int batch_size) : batch_size_(batch_size) {}

    // SOA_EIGEN_COLUMN
    template <typename SoALayout, typename TSoAParamsImpl, typename... Others>
      requires(SameValueType<TSoAParamsImpl, Others...> && TSoAParamsImpl::columnType == SoAColumnType::eigen)
    void register_tensor(const std::string& name,
                         int batch_size,
                         std::tuple<TSoAParamsImpl, cms::soa::size_type> column,
                         std::tuple<Others, cms::soa::size_type>... others) {
      using data_t = typename TSoAParamsImpl::ScalarType;
      auto [ptr, stride] = std::get<0>(column).tupleOrPointer();
      int n_elems = getElementsPerColumn(batch_size, SoALayout::alignment, sizeof(data_t));
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

      emplace_tensor<data_t>(name, SoALayout::alignment, ptr, batch_size, tensor_dims);
    }

    // SOA_EIGEN_COLUMN with default batch size
    template <typename SoALayout, typename TSoAParamsImpl, typename... Others>
      requires(SameValueType<TSoAParamsImpl, Others...> && TSoAParamsImpl::columnType == SoAColumnType::eigen)
    void register_tensor(const std::string& name,
                         std::tuple<TSoAParamsImpl, cms::soa::size_type> column,
                         std::tuple<Others, cms::soa::size_type>... others) {
      register_tensor<SoALayout, TSoAParamsImpl, Others...>(name, batch_size_, column, others...);
    }

    // SOA_COLUMN
    template <typename SoALayout, typename TSoAParamsImpl, typename... Others>
      requires(SameScalarType<TSoAParamsImpl, Others...> && TSoAParamsImpl::columnType == SoAColumnType::column)
    void register_tensor(const std::string& name,
                         int batch_size,
                         std::tuple<TSoAParamsImpl, cms::soa::size_type> column,
                         std::tuple<Others, cms::soa::size_type>... others) {
      using data_t = typename TSoAParamsImpl::ScalarType;
      int n_elems = getElementsPerColumn(batch_size, SoALayout::alignment, sizeof(data_t));
      assert_location(n_elems, std::get<0>(column).tupleOrPointer(), std::get<0>(others).tupleOrPointer()...);
      emplace_tensor<data_t>(name,
                             SoALayout::alignment,
                             std::get<0>(column).tupleOrPointer(),
                             batch_size,
                             std::vector<int>{1 + sizeof...(Others)});
    }

    // SOA_COLUMN with default batch size
    template <typename SoALayout, typename TSoAParamsImpl, typename... Others>
      requires(SameScalarType<TSoAParamsImpl, Others...> && TSoAParamsImpl::columnType == SoAColumnType::column)
    void register_tensor(const std::string& name,
                         std::tuple<TSoAParamsImpl, cms::soa::size_type> column,
                         std::tuple<Others, cms::soa::size_type>... others) {
      register_tensor<SoALayout, TSoAParamsImpl, Others...>(name, batch_size_, column, others...);
    }

    // SOA_SCALAR
    template <typename SoALayout, SoAColumnType column_t, typename T>
      requires(std::is_arithmetic_v<T> && column_t == SoAColumnType::scalar)
    void register_tensor(const std::string& name,
                         int batch_size,
                         std::tuple<SoAParametersImpl<column_t, T>, cms::soa::size_type> column) {
      const auto* ptr = std::get<0>(column).tupleOrPointer();
      emplace_tensor<T>(name, SoALayout::alignment, ptr, batch_size);
    }

    // SOA_SCALAR with default batch size
    template <typename SoALayout, SoAColumnType column_t, typename T>
      requires(std::is_arithmetic_v<T> && column_t == SoAColumnType::scalar)
    void register_tensor(const std::string& name,
                         std::tuple<SoAParametersImpl<column_t, T>, cms::soa::size_type> column) {
      register_tensor<SoALayout, column_t, T>(name, batch_size_, column);
    }

    // The order is defined by the order `register_tensor()` is called.
    // It can be changed by passing a vector of the block names afterwards.
    void change_order(std::vector<std::string> order) {
      assert(order.size() == order_.size() &&
             "TensorRegistry::change_order: size mismatch, all blocks have to be mentioned.");
      order_ = std::move(order);
    }
    size_t size() const { return registry_.size(); }
    const PortableTensorHandle& operator[](const size_t index) const { return registry_.at(order_[index]); }

    template <typename TQueue>
      requires ::alpaka::isQueue<TQueue>
    void copyToHost(const TQueue& queue) {
      for (const auto& name : order_)
        registry_.at(name).copyToHost(queue);
      // explicit synchronize to ensure data is in place before inference
      alpaka::wait(queue);
    }

    template <typename TQueue>
      requires ::alpaka::isQueue<TQueue>
    void copyToDevice(const TQueue& queue) {
      for (const auto& name : order_)
        registry_.at(name).copyToDevice(queue);
      // no need to explicitly synchronize, rely on implicit synchronization mechanism in framework
    }

  private:
    template <typename T>
    void emplace_tensor(
        const std::string& name, size_t alignment, const void* ptr, int batch_size, std::vector<int> dims = {}) {
      registry_.try_emplace(name, alignment, sizeof(T), ptr, get_type<T>(), batch_size, std::move(dims));
      order_.push_back(name);
    }

    int batch_size_;
    std::vector<std::string> order_;
    std::map<std::string, PortableTensorHandle> registry_;
  };

}  // namespace cms::torch::alpakatools

#endif  // PhysicsTools_PyTorchAlpaka_interface_TensorRegistry_h
