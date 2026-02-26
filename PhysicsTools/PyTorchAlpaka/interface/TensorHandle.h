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
    requires alpaka::isQueue<TQueue>
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
    requires alpaka::isQueue<TQueue>
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
    virtual void copy(TQueue& queue, const cms::torch::alpakatools::detail::MemcpyKind kind) = 0;
    virtual void* data() = 0;
  };

  // TODO: handle case when user register only one column:
  // e.g. .register_tensor("test", soa.pt()); (stride should be [1] instead of e.g. [1, 32])
  template <typename TQueue, typename T>
    requires alpaka::isQueue<TQueue>
  class TensorHandle : public ITensorHandle<TQueue> {
  public:
    explicit TensorHandle(const size_t alignment,
                          const size_t bytes,
                          T* data,
                          const int batch_size,
                          const std::vector<int> dims,
                          const bool is_scalar = false)
        : alignment_(alignment),
          bytes_(bytes),
          data_(data),
          dims_(batch_size, dims, is_scalar),
          policy_(data, dims_.volume() * num_elements_per_column(batch_size, alignment, bytes)) {
      init_sizes();
      init_strides();
    }

    size_t alignment() const override { return alignment_; }
    size_t bytes() const override { return bytes_; }
    ::torch::ScalarType type() const override { return get_type<T>(); }

    std::vector<long int> strides() const override { return strides_; }
    std::vector<long int> sizes() const override { return sizes_; }

    // propagate iterator from Dims
    using iterator_t = std::vector<int>::const_iterator;
    iterator_t begin() const { return dims_.begin(); }
    iterator_t end() const { return dims_.end(); }
    iterator_t cbegin() const { return dims_.cbegin(); }
    iterator_t cend() const { return dims_.cend(); }

  private:
    void copy(TQueue& queue, const cms::torch::alpakatools::detail::MemcpyKind kind) override {
      policy_.copy(queue, kind);
    }
    void* data() override { return static_cast<void*>(policy_.data()); }
    void init_sizes() {
      sizes_ = std::vector<long int>(dims_.size() + 1);
      sizes_[0] = dims_.batch_size();
      std::copy(dims_.begin(), dims_.end(), sizes_.begin() + 1);
      if (dims_.size() > 1 && dims_[0] == 1 && !dims_.is_scalar()) {
        sizes_.erase(sizes_.begin() + 1);
      }
    }

    void init_strides() {
      int N = dims_.size() + 1;
      strides_ = std::vector<long int>(N);

      int per_bunch = alignment_ / bytes_;
      int bunches = std::ceil(1.0 * dims_.batch_size() / per_bunch);

      // base stride initialization
      if (!dims_.is_scalar())
        strides_[0] = 1;
      else {
        // no tensor dimensions (scalar case)
        strides_[0] = 0;
        bunches = 1;
      }

      // stride for the second dimension (or first available)
      int stride_index = std::min(2, N - 1);
      strides_[stride_index] = bunches * per_bunch;

      // column-major layout (Eigen style)
      if (N > 2) {
        for (int i = 3; i < N; ++i)
          strides_[i] = strides_[i - 1] * dims_[i - 2];
        // stride for the "batch" dimension
        strides_[1] = strides_[N - 1] * dims_[N - 2];
        // 1D column
        if (dims_[0] == 1)
          strides_.erase(strides_.begin() + 1);
      }
    }

    const size_t alignment_;
    const size_t bytes_;
    T* data_;
    const Dims dims_;

    std::vector<long int> strides_;
    std::vector<long int> sizes_;

    // workaround until pytorch COW Tensors is implemented
    // in the mainstream framework or cmssw add patch with COW inital state.
    cms::torch::alpakatools::detail::Policy<TQueue, T> policy_;
  };

}  // namespace cms::torch::alpakatools::detail

#endif  // PhysicsTools_PyTorchAlpaka_interface_TensorHandle_h
