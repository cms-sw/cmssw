#ifndef PhysicsTools_PyTorchAlpaka_interface_TensorHandle_h
#define PhysicsTools_PyTorchAlpaka_interface_TensorHandle_h

#include <cmath>
#include <numeric>
#include <vector>

#include "PhysicsTools/PyTorch/interface/TorchInterface.h"
#include "PhysicsTools/PyTorchAlpaka/interface/Policy.h"

namespace cms::torch::alpakatools {

  inline int getElementsPerColumn(const int n_elems, const size_t alignment, const size_t bytes) {
    int per_bunch = alignment / bytes;
    int bunches = std::ceil(1.0 * n_elems / per_bunch);
    return bunches * per_bunch;
  }

  class Dims {
  public:
    explicit Dims(const int batch_size, const std::vector<int> dims) : batch_size_(batch_size), dims_(dims) {}
    int operator[](size_t idx) const { return dims_[idx]; }
    size_t size() const { return dims_.size(); }
    bool empty() const { return dims_.empty(); }
    int batch_size() const { return batch_size_; }
    int dim0() const { return dims_.empty() ? 1 : dims_[0]; }
    int n_elems() const { return std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>()); }
    int n_elems_per_batch() const { return n_elems() / batch_size_; }

    std::vector<int> shape() const {
      std::vector<int> s(1, batch_size_);
      s.insert(s.end(), dims_.begin(), dims_.end());
      return s;
    }

    // iterator
    using iterator_t = std::vector<int>::const_iterator;
    iterator_t begin() const { return dims_.begin(); }
    iterator_t end() const { return dims_.end(); }
    iterator_t cbegin() const { return dims_.cbegin(); }
    iterator_t cend() const { return dims_.cend(); }

  private:
    int batch_size_;
    std::vector<int> dims_;
  };

  template <typename TPolicy>
  class TensorHandle {
  public:
    explicit TensorHandle(const size_t alignment,
                          const size_t bytes,
                          const void* data,
                          const ::torch::ScalarType type,
                          const int batch_size,
                          const std::vector<int> dims)
        : alignment_(alignment),
          bytes_(bytes),
          data_(data),
          dims_(batch_size, dims),
          type_(type),
          policy_(data, dims_.dim0() * getElementsPerColumn(batch_size, alignment, bytes) * bytes) {
      init_strides();
      init_sizes();
    }

    size_t alignment() const { return alignment_; }
    size_t bytes() const { return bytes_; }
    const void* data() const { return data_; }
    ::torch::ScalarType type() const { return type_; }

    std::vector<int> shape() const { return dims_.shape(); }
    const std::vector<long int>& strides() const { return strides_; }
    const std::vector<long int>& sizes() const { return sizes_; }

    template <typename TQueue>
      requires ::alpaka::isQueue<TQueue>
    void copyToHost(const TQueue& queue) {
      policy_.copyToHost(queue);
    }

    template <typename TQueue>
      requires ::alpaka::isQueue<TQueue>
    void copyToDevice(const TQueue& queue) {
      policy_.copyToDevice(queue);
    }

  private:
    void init_sizes() {
      sizes_ = std::vector<long int>(dims_.size() + 1);
      sizes_[0] = dims_.batch_size();
      std::copy(dims_.begin(), dims_.end(), sizes_.begin() + 1);
      if (dims_.size() > 1 && dims_[0] == 1) {
        sizes_.erase(sizes_.begin() + 1);
      }
    }

    void init_strides() {
      int N = dims_.size() + 1;
      strides_ = std::vector<long int>(N);

      int per_bunch = alignment_ / bytes_;
      int bunches = std::ceil(1.0 * dims_.batch_size() / per_bunch);

      // base stride initialization
      if (!dims_.empty())
        strides_[0] = 1;
      else {
        // No tensor dimensions: scalar case
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
        // 1D column optimization
        if (dims_[0] == 1)
          strides_.erase(strides_.begin() + 1);
      }
    }

    const size_t alignment_;
    const size_t bytes_;
    const void* data_;
    const Dims dims_;
    const ::torch::ScalarType type_;

    std::vector<long int> strides_;
    std::vector<long int> sizes_;

    TPolicy policy_;
  };

#ifndef ALPAKA_ACC_GPU_HIP_ENABLED
  using PortableTensorHandle = TensorHandle<HipPolicy>;
#else
  using PortableTensorHandle = TensorHandle<DefaultPolicy>;
#endif

}  // namespace cms::torch::alpakatools

#endif  // PhysicsTools_PyTorchAlpaka_interface_TensorHandle_h
