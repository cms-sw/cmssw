#ifndef PhysicsTools_PyTorchAlpaka_interface_TensorHandle_h
#define PhysicsTools_PyTorchAlpaka_interface_TensorHandle_h

#include <cmath>
#include <numeric>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "PhysicsTools/PyTorch/interface/TorchInterface.h"
#include "PhysicsTools/PyTorchAlpaka/interface/Policy.h"

// Forward declaration for friend
namespace cms::torch::alpakatools {
  template <typename TQueue>
    requires alpaka::concepts::Queue<TQueue>
  class TensorCollection;
}

namespace cms::torch::alpakatools::detail {

  template <typename T>
  ::torch::ScalarType get_type() {
    return ::torch::CppTypeToScalarType<std::remove_const_t<T>>();
  }

  inline int num_elements_per_column(const int n_elems, const size_t alignment, const size_t bytes) {
    int per_bunch = alignment / bytes;
    int bunches = (n_elems + per_bunch - 1) / per_bunch;
    return bunches * per_bunch;
  }

  class Dims {
  public:
    explicit Dims(const int batch_size, const std::vector<int> dims, const bool is_scalar = false)
        : batch_size_(batch_size), dims_(dims), is_scalar_{is_scalar} {}
    int operator[](size_t idx) const { return dims_[idx]; }
    size_t size() const { return dims_.size(); }
    bool empty() const { return dims_.empty(); }
    bool is_scalar() const { return is_scalar_; }
    int batch_size() const { return batch_size_; }
    int volume() const { return std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>()); }

    // iterator
    using iterator_t = std::vector<int>::const_iterator;
    iterator_t begin() const { return dims_.begin(); }
    iterator_t end() const { return dims_.end(); }
    iterator_t cbegin() const { return dims_.cbegin(); }
    iterator_t cend() const { return dims_.cend(); }

  private:
    int batch_size_;
    std::vector<int> dims_;
    bool is_scalar_;
  };

  template <typename TQueue>
    requires alpaka::concepts::Queue<TQueue>
  class ITensorHandle {
  public:
    virtual ~ITensorHandle() = default;

    virtual size_t alignment() const = 0;
    virtual size_t bytes() const = 0;
    virtual ::torch::ScalarType type() const = 0;

    virtual std::vector<long int> sizes() const = 0;
    virtual std::vector<long int> strides() const = 0;

    template <typename TQueue_T>
    friend ::torch::Tensor arrayToTensor(::torch::Device device, ITensorHandle<TQueue_T>& tensor_handle);
    friend class ::cms::torch::alpakatools::TensorCollection<TQueue>;

  private:
    virtual void copy(TQueue& queue) = 0;
    virtual void* data() = 0;
  };

  // helper to construct the CopyLayout struct
  inline CopyLayout copy_layout(const Dims& dims, const int total_size, const size_t alignment, const size_t bytes) {
    const bool scalar = dims.is_scalar();

    const int source_rows = scalar ? 1 : total_size;
    const int copied_rows = scalar ? (dims.batch_size() == 0 ? 0 : 1) : dims.batch_size();

    return {.columns = scalar ? 1u : static_cast<size_t>(dims.volume()),
            .rows_per_column = static_cast<size_t>(copied_rows),
            .source_stride = static_cast<size_t>(num_elements_per_column(source_rows, alignment, bytes)),
            .destination_stride = static_cast<size_t>(num_elements_per_column(copied_rows, alignment, bytes))};
  }

  // TODO: handle case when user register only one column:
  // e.g. .register_tensor("test", soa.pt()); (stride should be [1] instead of e.g. [1, 32])
  template <typename TQueue, typename T>
    requires alpaka::concepts::Queue<TQueue>
  class TensorHandle : public ITensorHandle<TQueue> {
  public:
    explicit TensorHandle(const size_t alignment,
                          const size_t bytes,
                          T* data,
                          const int batch_size,
                          const int total_size,
                          const std::vector<int> dims,
                          const bool is_scalar = false)
        : alignment_(alignment),
          bytes_(bytes),
          data_(data),
          total_size_(total_size),
          dims_(batch_size, dims, is_scalar),
          policy_(data, copy_layout(dims_, total_size_, alignment_, bytes_)) {
      init_sizes();
    }

    size_t alignment() const override { return alignment_; }
    size_t bytes() const override { return bytes_; }
    ::torch::ScalarType type() const override { return get_type<T>(); }

    std::vector<long int> strides() const override { return get_exposed_strides(); }
    std::vector<long int> sizes() const override { return sizes_; }

    // propagate iterator from Dims
    using iterator_t = std::vector<int>::const_iterator;
    iterator_t begin() const { return dims_.begin(); }
    iterator_t end() const { return dims_.end(); }
    iterator_t cbegin() const { return dims_.cbegin(); }
    iterator_t cend() const { return dims_.cend(); }

  private:
    void copy(TQueue& queue) override { policy_.copy(queue); }
    void* data() override { return static_cast<void*>(policy_.data()); }
    void init_sizes() {
      sizes_ = std::vector<long int>(dims_.size() + 1);
      sizes_[0] = dims_.batch_size();
      std::copy(dims_.begin(), dims_.end(), sizes_.begin() + 1);
      if (dims_.size() > 1 && dims_[0] == 1 && !dims_.is_scalar()) {
        sizes_.erase(sizes_.begin() + 1);
      }
    }

    std::vector<long int> get_exposed_strides() const {
      const auto is_const_view = std::is_const_v<T>;

      const int N = dims_.size() + 1;
      auto exposed_strides = std::vector<long int>(N);

      // base stride initialization
      // no tensor dimensions in scalar case
      exposed_strides[0] = dims_.is_scalar() ? 0 : 1;

      // stride for the second dimension (or first available)
      // If the view is constant --> a copy is triggered --> use destination_stride
      // If the view is mutable --> no copy triggered --> use source_stride
      int stride_index = std::min(2, N - 1);
      exposed_strides[stride_index] =
          is_const_view ? policy_.getCopyLayout().destination_stride : policy_.getCopyLayout().source_stride;

      // column-major layout (Eigen style)
      if (N > 2) {
        for (int i = 3; i < N; ++i)
          exposed_strides[i] = exposed_strides[i - 1] * dims_[i - 2];
        // stride for the "batch" dimension
        exposed_strides[1] = exposed_strides[N - 1] * dims_[N - 2];
        // 1D column
        if (dims_[0] == 1)
          exposed_strides.erase(exposed_strides.begin() + 1);
      }

      return exposed_strides;
    }

    const size_t alignment_;
    const size_t bytes_;
    T* data_;
    const int total_size_;
    const Dims dims_;

    std::vector<long int> sizes_;

    // workaround until pytorch COW Tensors is implemented
    // in the mainstream framework or cmssw add patch with COW inital state.
    cms::torch::alpakatools::detail::Policy<TQueue, T> policy_;
  };

}  // namespace cms::torch::alpakatools::detail

#endif  // PhysicsTools_PyTorchAlpaka_interface_TensorHandle_h
