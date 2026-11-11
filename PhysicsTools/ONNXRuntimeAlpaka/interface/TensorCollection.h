#ifndef PhysicsTools_ONNXRuntimeAlpaka_interface_TensorCollection_h
#define PhysicsTools_ONNXRuntimeAlpaka_interface_TensorCollection_h

// Adapted from PhysicsTools/PyTorchAlpaka/interface/TensorCollection.h .
// The interface to register the SoA columns is the same, including TensorSlice for mini-batches; in addition, the
// layout used to present each tensor to ONNX Runtime can be changed with set_layout().
// Unlike PyTorchAlpaka, an SOA_SCALAR is presented as a one-element tensor of shape [1], independently of the size of
// the SoA and of the mini-batches.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "PhysicsTools/ONNXRuntimeAlpaka/interface/Exception.h"
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

  // The columns of a tensor must be consecutive columns of the same SoA, in the order of the layout. This depends only
  // on the SoA layout and on the code that registers the tensor, so it is checked with an assertion.
  template <typename T, typename... Others>
  void assert_location(int elements, const T* column, const Others*... others) {
    bool contiguous = check_location(elements, column, others...);
    assert(contiguous && "Tensor columns are not contiguous in memory!");
  }

  template <typename T, typename... Others>
  concept SameTypes = (std::same_as<T, Others> && ...);

  template <typename TSoAParamsImpl, typename... Others>
  concept SameValueType = SameTypes<typename TSoAParamsImpl::ValueType, typename Others::ValueType...>;

  template <typename TSoAParamsImpl, typename... Others>
  concept SameScalarType = SameTypes<typename TSoAParamsImpl::ScalarType, typename Others::ScalarType...>;

  // Selects the rows of an SoA used for a tensor: all of them (the default), or the mini-batch batch_id of size
  // batch_size. The last mini-batch can contain fewer rows.
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
      if (batch_size_ == 0)
        detail::throwException("TensorCollection", "TensorSlice: batch size must be greater than zero.");

      if (batch_id_ > (total_size - 1u) / batch_size_) {
        detail::throwException("TensorCollection",
                               "TensorSlice: batch id " + std::to_string(batch_id_) +
                                   " is out of bounds for total size " + std::to_string(total_size) +
                                   " and batch size " + std::to_string(batch_size_) + ".");
      }
      const auto offset = batch_id_ * batch_size_;
      return {.offset = offset, .size = std::min(batch_size_, total_size - offset)};
    }

  private:
    uint32_t batch_id_ = 0;
    uint32_t batch_size_ = 0;
    bool full_ = true;
  };

  // Register SoA memory blocks to be presented to ONNX Runtime as tensors.
  // The shape of each tensor is derived from its SoA and from an optional TensorSlice, so tensors from SoAs with a
  // different number of elements can be used together.
  //
  // Provided memory blocks must be of the same type and to be contiguous e.g.:
  //
  // GENERATE_SOA_LAYOUT(ParticleLayout,
  //                     SOA_COLUMN(float, pt),
  //                     SOA_COLUMN(float, eta),
  //                     SOA_COLUMN(float, phi),
  //                     SOA_SCALAR(float, scale))
  //
  // can register the following:
  //
  // TensorCollection<Queue> registry;
  // registry.add<ParticleLayout>("features", records.pt(), records.eta(), records.phi());
  // registry.add<ParticleLayout>("scale", records.scale());
  //
  // By default the "features" tensor is presented to ONNX Runtime with the shape [size, 3], packing the three columns
  // into a temporary buffer. To avoid the copy, the SoA memory can be used directly as a [3, padded size] tensor, if
  // the model has been exported to accept a transposed input:
  //
  // registry.set_layout("features", Layout::FeatureMajor);
  //
  // An SOA_SCALAR is presented as a tensor of shape [1], and is used in place. If the model needs it for each row, it
  // should broadcast it explicitly.
  //
  // To run the inference in mini-batches, pass a TensorSlice to select the rows of each batch; the add() function
  // computes the offset of the batch and the number of rows in it, and ensures the provided columns are contiguous in
  // memory:
  //
  // for (uint32_t batch_id = 0; batch_id < n_batches; ++batch_id) {
  //   TensorCollection<Queue> registry;
  //   registry.add<ParticleLayout>("features", TensorSlice{batch_id, batch_size}, records.pt(), records.eta(), records.phi());
  //   registry.add<ParticleLayout>("scale", records.scale());
  //   ...
  // }

  template <typename TQueue>
    requires alpaka::concepts::Queue<TQueue>
  class TensorCollection {
  public:
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

      int n_elems = detail::num_elements_per_column(soa_size, SoALayout::alignment, sizeof(DataType));
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

      emplace_tensor(name, SoALayout::alignment, ptr, effective_size, soa_size, offset, tensor_dims);
    }

    // SOA_EIGEN_COLUMN without slicing: use all rows.
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

      int n_elems = detail::num_elements_per_column(soa_size, SoALayout::alignment, sizeof(DataType));
      assert_location(n_elems, ptr, std::get<0>(others).data()...);

      ptr += offset;
      emplace_tensor(name, SoALayout::alignment, ptr, effective_size, soa_size, offset, {1 + sizeof...(Others)});
    }

    // SOA_COLUMN without slicing: use all rows.
    template <typename SoALayout, typename TSoAParamsImpl, typename... Others>
      requires(SameScalarType<TSoAParamsImpl, Others...> &&
               TSoAParamsImpl::columnType == cms::soa::SoAColumnType::column)
    void add(const std::string& name,
             std::tuple<TSoAParamsImpl, cms::soa::size_type> column,
             std::tuple<Others, cms::soa::size_type>... others) {
      add<SoALayout, TSoAParamsImpl, Others...>(name, TensorSlice{}, column, others...);
    }

    // SOA_SCALAR: a single value, presented as a tensor of shape [1] independently of the SoA size and of the
    // mini-batches, so there is no TensorSlice overload.
    template <typename SoALayout, cms::soa::SoAColumnType column_t, typename T>
      requires(std::is_arithmetic_v<T> && column_t == cms::soa::SoAColumnType::scalar)
    void add(const std::string& name,
             std::tuple<cms::soa::SoAParametersImpl<column_t, T>, cms::soa::size_type> column) {
      auto ptr = std::get<0>(column).data();
      emplace_tensor(name, SoALayout::alignment, ptr, 1, 1, 0, {1}, true);
    }

    // The order is defined by the order `add()` is called. It is used to match the tensors to the model inputs or
    // outputs only if their names do not match the names used in the model, see match().
    // It can be changed by passing a vector of the block names afterwards.
    void change_order(std::vector<std::string> order) {
      if (order.size() != order_.size())
        detail::throwException("TensorCollection",
                               "TensorCollection::change_order: size mismatch, all tensors must be specified.");
      for (size_t i = 0; i < order.size(); ++i) {
        if (not registry_.contains(order[i]))
          detail::throwException("TensorCollection",
                                 "TensorCollection::change_order: tensor name '" + order[i] + "' is not registered.");
        if (std::find(order.begin(), order.begin() + i, order[i]) != order.begin() + i)
          detail::throwException("TensorCollection",
                                 "TensorCollection::change_order: tensor name '" + order[i] + "' is repeated.");
      }
      order_ = std::move(order);
    }

    // Match the tensors to the inputs or outputs of a model, given their names, and return for each of them the index
    // of the corresponding tensor:
    //   - if the names of all the tensors are names of the model inputs (outputs), they are matched by name,
    //     independently of the order in which they have been added;
    //   - if none of them is, they are matched by position, in the order in which they have been added, or set by
    //     change_order();
    //   - if only some of them are, the names are most likely wrong, and an exception is thrown.
    // `kind` and `model` are used only in the error messages, e.g. "input" and the path of the model.
    std::vector<size_t> match(const std::vector<std::string>& names,
                              const std::string& kind,
                              const std::string& model) const {
      if (names.size() != order_.size()) {
        detail::throwException("TensorCollection",
                               "The model " + model + " has " + std::to_string(names.size()) + " " + kind + "s (" +
                                   join(names) + "), but " + std::to_string(order_.size()) + " " + kind +
                                   " tensors were provided (" + join(order_) + ").");
      }
      std::vector<size_t> index(names.size());
      const auto found = std::count_if(
          names.begin(), names.end(), [this](const std::string& name) { return registry_.contains(name); });
      if (found == static_cast<std::ptrdiff_t>(names.size())) {
        // match by name
        for (size_t i = 0; i < names.size(); ++i)
          index[i] = std::find(order_.begin(), order_.end(), names[i]) - order_.begin();
      } else if (found == 0) {
        // match by position
        std::iota(index.begin(), index.end(), 0);
      } else {
        detail::throwException("TensorCollection",
                               "The names of the " + kind + " tensors (" + join(order_) +
                                   ") match only some of the names of the " + kind + "s of the model " + model + " (" +
                                   join(names) +
                                   "): use the same names to match them by name, or names different from all of "
                                   "them to match them by position.");
      }
      return index;
    }

    // Change the layout used to present a tensor to ONNX Runtime.
    void set_layout(const std::string& name, Layout layout) {
      auto it = registry_.find(name);
      if (it == registry_.end())
        detail::throwException("TensorCollection",
                               "TensorCollection::set_layout: tensor name '" + name + "' is not registered.");
      it->second->set_layout(layout);
    }

    size_t size() const { return registry_.size(); }
    bool contains(const std::string& name) const { return registry_.contains(name); }
    const std::string& name(const size_t index) const { return order_[index]; }
    detail::ITensorHandle& operator[](const size_t index) const { return *registry_.at(order_[index]); }

  private:
    static std::string join(const std::vector<std::string>& names) {
      std::string result;
      for (const auto& name : names)
        result += (result.empty() ? "'" : ", '") + name + "'";
      return result;
    }

    // propagate pointer (Tptr) and type to distinguish between T* and const T*
    template <typename Tptr>
    void emplace_tensor(const std::string& name,
                        size_t alignment,
                        Tptr ptr,
                        int batch_size,
                        int total_size,
                        int offset,
                        std::vector<int> dims = {1},
                        const bool is_scalar = false) {
      using T = std::remove_pointer_t<Tptr>;

      if (registry_.contains(name))
        detail::throwException("TensorCollection", "tensor name '" + name + "' is already registered.");

      const bool full_range = (offset == 0 and batch_size == total_size);
      registry_.emplace(name,
                        std::make_unique<detail::TensorHandle<T>>(
                            alignment, sizeof(T), ptr, batch_size, total_size, full_range, std::move(dims), is_scalar));
      order_.push_back(name);
    }

    std::vector<std::string> order_;
    std::unordered_map<std::string, std::unique_ptr<detail::ITensorHandle>> registry_;
  };

}  // namespace cms::Ort::alpakatools

#endif  // PhysicsTools_ONNXRuntimeAlpaka_interface_TensorCollection_h
