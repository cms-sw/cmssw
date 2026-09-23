#ifndef PhysicsTools_ONNXRuntimeAlpaka_interface_TensorCollection_h
#define PhysicsTools_ONNXRuntimeAlpaka_interface_TensorCollection_h

// Adapted from PhysicsTools/PyTorchAlpaka/interface/TensorCollection.h .
// The interface to register the SoA columns is unchanged; in addition, the layout used to present each tensor to
// ONNX Runtime can be changed with set_layout().

#include <cassert>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "PhysicsTools/ONNXRuntimeAlpaka/interface/TensorHandle.h"

namespace cms::Ort::alpakatools {

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

  // Container for user defined memory blobs that will be converted to ONNX Runtime tensors, constructed directly from
  // the provided recipes and contiguous memory blocks.
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
  // TensorCollection<Queue> registry(batch_size, total_size);
  // registry.add<ParticleLayout>("features", batch_id, records.pt(), records.eta(), records.phi());
  //
  // In the above example, the add function automatically computes the offset for the batch and ensures the provided
  // columns are contiguous in memory. To perform the inference on the entire dataset without batching, simply pass
  // just the total size:
  //
  // TensorCollection<Queue> registry(total_size);
  // registry.add<ParticleLayout>("features", records.pt(), records.eta(), records.phi());
  //
  // By default the tensor is presented to ONNX Runtime with the shape [total_size, 3], packing the three columns into
  // a temporary buffer. To avoid the copy, the SoA memory can be used directly as a [3, padded size] tensor, if the
  // model has been exported to accept a transposed input:
  //
  // registry.set_layout("features", Layout::FeatureMajor);

  template <typename TQueue>
    requires(alpaka::isQueue<TQueue>)
  class TensorCollection {
  public:
    explicit TensorCollection(int total_size) : batch_size_(total_size), total_size_(total_size) { assert_sizes(); }
    explicit TensorCollection(int batch_size, int total_size) : batch_size_(batch_size), total_size_(total_size) {
      assert_sizes();
    }

    // SOA_EIGEN_COLUMN
    template <typename SoALayout, typename TSoAParamsImpl, typename... Others>
      requires(SameValueType<TSoAParamsImpl, Others...> && TSoAParamsImpl::columnType == cms::soa::SoAColumnType::eigen)
    void add(const std::string& name,
             int batch_id,
             std::tuple<TSoAParamsImpl, cms::soa::size_type> column,
             std::tuple<Others, cms::soa::size_type>... others) {
      using DataType = typename TSoAParamsImpl::ScalarType;
      assert_batch_id(batch_id);
      int offset = batch_id * batch_size_;
      auto ptr = std::get<0>(column).data();
      int n_elems = detail::num_elements_per_column(total_size_, SoALayout::alignment, sizeof(DataType));
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

      // Handle the case in which the last batch contains less elements
      auto effective_batch_size = std::min(batch_size_, total_size_ - offset);
      emplace_tensor(name, SoALayout::alignment, ptr, effective_batch_size, offset, tensor_dims);
    }

    // SOA_EIGEN_COLUMN with default batch size = default size
    template <typename SoALayout, typename TSoAParamsImpl, typename... Others>
      requires(SameValueType<TSoAParamsImpl, Others...> && TSoAParamsImpl::columnType == cms::soa::SoAColumnType::eigen)
    void add(const std::string& name,
             std::tuple<TSoAParamsImpl, cms::soa::size_type> column,
             std::tuple<Others, cms::soa::size_type>... others) {
      add<SoALayout, TSoAParamsImpl, Others...>(name, 0, column, others...);
    }

    // SOA_COLUMN
    template <typename SoALayout, typename TSoAParamsImpl, typename... Others>
      requires(SameScalarType<TSoAParamsImpl, Others...> &&
               TSoAParamsImpl::columnType == cms::soa::SoAColumnType::column)
    void add(const std::string& name,
             int batch_id,
             std::tuple<TSoAParamsImpl, cms::soa::size_type> column,
             std::tuple<Others, cms::soa::size_type>... others) {
      using DataType = typename TSoAParamsImpl::ScalarType;
      assert_batch_id(batch_id);
      int offset = batch_id * batch_size_;
      auto ptr = std::get<0>(column).data();
      int n_elems = detail::num_elements_per_column(total_size_, SoALayout::alignment, sizeof(DataType));
      assert_location(n_elems, ptr, std::get<0>(others).data()...);

      ptr += offset;
      auto effective_batch_size = std::min(batch_size_, total_size_ - offset);
      emplace_tensor(name, SoALayout::alignment, ptr, effective_batch_size, offset, {1 + sizeof...(Others)});
    }

    // SOA_COLUMN with default batch size = total size
    template <typename SoALayout, typename TSoAParamsImpl, typename... Others>
      requires(SameScalarType<TSoAParamsImpl, Others...> &&
               TSoAParamsImpl::columnType == cms::soa::SoAColumnType::column)
    void add(const std::string& name,
             std::tuple<TSoAParamsImpl, cms::soa::size_type> column,
             std::tuple<Others, cms::soa::size_type>... others) {
      add<SoALayout, TSoAParamsImpl, Others...>(name, 0, column, others...);
    }

    // SOA_SCALAR
    template <typename SoALayout, cms::soa::SoAColumnType column_t, typename T>
      requires(std::is_arithmetic_v<T> && column_t == cms::soa::SoAColumnType::scalar)
    void add(const std::string& name,
             std::tuple<cms::soa::SoAParametersImpl<column_t, T>, cms::soa::size_type> column) {
      auto ptr = std::get<0>(column).data();
      emplace_tensor(name, SoALayout::alignment, ptr, batch_size_, 0, {1}, true);
    }

    // The order is defined by the order `add()` is called, and must match the order of the model inputs or outputs.
    // It can be changed by passing a vector of the block names afterwards.
    void change_order(std::vector<std::string> order) {
      assert(order.size() == order_.size() &&
             "TensorCollection::change_order: size mismatch, all blocks have to be mentioned.");
      order_ = std::move(order);
    }

    // Change the layout used to present a tensor to ONNX Runtime.
    void set_layout(const std::string& name, Layout layout) { registry_.at(name)->set_layout(layout); }

    size_t size() const { return registry_.size(); }
    const std::string& name(const size_t index) const { return order_[index]; }
    detail::ITensorHandle& operator[](const size_t index) const { return *registry_.at(order_[index]); }

  private:
    // propagate pointer (Tptr) and type to distinguish between T* and const T*
    template <typename Tptr>
    void emplace_tensor(const std::string& name,
                        size_t alignment,
                        Tptr ptr,
                        int batch_size,
                        int offset,
                        std::vector<int> dims = {1},
                        const bool is_scalar = false) {
      using T = std::remove_pointer_t<Tptr>;
      const bool full_range = (offset == 0 and batch_size == total_size_);
      registry_.try_emplace(
          name,
          std::make_unique<detail::TensorHandle<T>>(
              alignment, sizeof(T), ptr, batch_size, total_size_, full_range, std::move(dims), is_scalar));
      order_.push_back(name);
    }

    void assert_sizes() {
      assert(total_size_ >= 0 && "Total size must be positive!");
      if (batch_size_ == 0) {
        assert(total_size_ == 0 && "Batch size 0 only allowed when total size is 0");
        return;
      }
      assert(batch_size_ > 0 && "Batch size must be positive!");
    }

    void assert_batch_id(int batch_id) {
      assert(batch_id >= 0 && "Batch id must be non-negative!");
      assert((total_size_ == 0 || (batch_id * batch_size_ < total_size_)) && "Batch id is out of bounds!");
    }

    int batch_size_;
    int total_size_;
    std::vector<std::string> order_;
    std::unordered_map<std::string, std::unique_ptr<detail::ITensorHandle>> registry_;
  };

}  // namespace cms::Ort::alpakatools

#endif  // PhysicsTools_ONNXRuntimeAlpaka_interface_TensorCollection_h
