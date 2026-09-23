#ifndef PhysicsTools_ONNXRuntimeAlpaka_interface_TensorHandle_h
#define PhysicsTools_ONNXRuntimeAlpaka_interface_TensorHandle_h

// Adapted from PhysicsTools/PyTorchAlpaka/interface/TensorHandle.h .
// The SoA metadata logic (Dims, padding, sizes and strides) is unchanged; the PyTorch specific parts are replaced by
// ONNX Runtime ones:
//   - the element type is an ONNXTensorElementDataType;
//   - there is no Policy: ONNX Runtime never writes to its inputs, so const data does not need to be copied;
//   - each tensor has a Layout and can describe how to pack it into a contiguous, row-major buffer, because
//     ONNX Runtime does not support strided tensors.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <numeric>
#include <type_traits>
#include <vector>

#include "onnxruntime/onnxruntime_cxx_api.h"

#include "PhysicsTools/ONNXRuntimeAlpaka/interface/PackDescriptor.h"

namespace cms::Ort::alpakatools {

  // How a tensor is presented to ONNX Runtime.
  //   - SampleMajor: the tensor has the same shape as in PyTorch, e.g. [batch, features]. If the data spans more than a
  //     single SoA column, it is packed into (and unpacked from) a temporary contiguous buffer.
  //   - FeatureMajor: the SoA memory is used directly, as a [columns, padded size] tensor. No copy is done, but the
  //     model must accept (and produce) transposed tensors, and it will also process the padding elements.
  //     It is supported only when the tensor spans the whole collection (no mini-batches) and is not a scalar.
  enum class Layout { SampleMajor, FeatureMajor };

}  // namespace cms::Ort::alpakatools

namespace cms::Ort::alpakatools::detail {

  template <typename T>
  ONNXTensorElementDataType get_type() {
    return ::Ort::TypeToTensorType<std::remove_const_t<T>>::type;
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

  class ITensorHandle {
  public:
    virtual ~ITensorHandle() = default;

    virtual size_t alignment() const = 0;
    virtual size_t bytes() const = 0;
    virtual ONNXTensorElementDataType type() const = 0;

    // shape and strides (in elements) as they would be seen by PyTorch, batch dimension first
    virtual std::vector<int64_t> sizes() const = 0;
    virtual std::vector<int64_t> strides() const = 0;

    // pointer to the first element of the tensor, i.e. of the first column in the current batch
    virtual void* data() const = 0;
    virtual bool is_scalar() const = 0;
    virtual int batch_size() const = 0;
    // true if the tensor spans all the elements of the collection, i.e. it is not a mini-batch
    virtual bool full_range() const = 0;
    // number of elements in each column, including the padding
    virtual int64_t padded_column_size() const = 0;

    Layout layout() const { return layout_; }
    void set_layout(Layout layout) {
      assert((layout == Layout::SampleMajor or (full_range() and not is_scalar())) &&
             "The FeatureMajor layout requires a non-scalar tensor spanning the whole collection.");
      layout_ = layout;
    }

    // number of elements per sample, i.e. the volume of the non-batch dimensions
    int64_t features() const {
      auto s = sizes();
      return std::accumulate(s.begin() + 1, s.end(), int64_t{1}, std::multiplies<int64_t>());
    }

    // A tensor made of a single SoA column is already contiguous in the SampleMajor layout.
    bool is_contiguous() const { return layout_ == Layout::FeatureMajor or (features() == 1 and not is_scalar()); }

    // shape of the tensor passed to ONNX Runtime
    std::vector<int64_t> shape() const {
      if (layout_ == Layout::FeatureMajor)
        return {features(), padded_column_size()};
      return sizes();
    }

    // size in bytes of the tensor passed to ONNX Runtime
    size_t shape_bytes() const {
      auto s = shape();
      return std::accumulate(s.begin(), s.end(), int64_t{1}, std::multiplies<int64_t>()) * bytes();
    }

    // describe how to pack the tensor into a contiguous row-major buffer
    PackDescriptor pack_descriptor() const {
      auto s = sizes();
      auto t = strides();
      assert(s.size() - 1 <= PackDescriptor::kMaxDims && "Too many tensor dimensions.");
      PackDescriptor desc;
      desc.elem_size = bytes();
      desc.batch = s[0];
      desc.batch_stride = t[0];
      desc.n_dims = s.size() - 1;
      for (int d = 0; d < desc.n_dims; ++d) {
        desc.sizes[d] = s[d + 1];
        desc.strides[d] = t[d + 1];
      }
      return desc;
    }

  private:
    Layout layout_ = Layout::SampleMajor;
  };

  template <typename T>
  class TensorHandle : public ITensorHandle {
  public:
    explicit TensorHandle(const size_t alignment,
                          const size_t bytes,
                          T* data,
                          const int batch_size,
                          const int total_size,
                          const bool full_range,
                          const std::vector<int> dims,
                          const bool is_scalar = false)
        : alignment_(alignment),
          bytes_(bytes),
          data_(data),
          total_size_(total_size),
          full_range_(full_range),
          dims_(batch_size, dims, is_scalar) {
      init_sizes();
      init_strides();
    }

    size_t alignment() const override { return alignment_; }
    size_t bytes() const override { return bytes_; }
    ONNXTensorElementDataType type() const override { return get_type<T>(); }

    std::vector<int64_t> strides() const override { return strides_; }
    std::vector<int64_t> sizes() const override { return sizes_; }

    // ONNX Runtime never modifies its inputs, so it is safe to cast away the constness
    void* data() const override { return const_cast<std::remove_const_t<T>*>(data_); }
    bool is_scalar() const override { return dims_.is_scalar(); }
    int batch_size() const override { return dims_.batch_size(); }
    bool full_range() const override { return full_range_; }
    int64_t padded_column_size() const override { return num_elements_per_column(total_size_, alignment_, bytes_); }

    // propagate iterator from Dims
    using iterator_t = std::vector<int>::const_iterator;
    iterator_t begin() const { return dims_.begin(); }
    iterator_t end() const { return dims_.end(); }
    iterator_t cbegin() const { return dims_.cbegin(); }
    iterator_t cend() const { return dims_.cend(); }

  private:
    void init_sizes() {
      sizes_ = std::vector<int64_t>(dims_.size() + 1);
      sizes_[0] = dims_.batch_size();
      std::copy(dims_.begin(), dims_.end(), sizes_.begin() + 1);
      if (dims_.size() > 1 && dims_[0] == 1 && !dims_.is_scalar()) {
        sizes_.erase(sizes_.begin() + 1);
      }
    }

    void init_strides() {
      int N = dims_.size() + 1;
      strides_ = std::vector<int64_t>(N);

      int per_bunch = alignment_ / bytes_;
      int bunches = (total_size_ + per_bunch - 1) / per_bunch;

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
    const int total_size_;
    const bool full_range_;
    const Dims dims_;

    std::vector<int64_t> strides_;
    std::vector<int64_t> sizes_;
  };

}  // namespace cms::Ort::alpakatools::detail

#endif  // PhysicsTools_ONNXRuntimeAlpaka_interface_TensorHandle_h
