#ifndef HeterogeneousCore_AlpakaInterface_interface_workdivision_h
#define HeterogeneousCore_AlpakaInterface_interface_workdivision_h

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/vec.h"

namespace cms::alpakatools {

  using namespace alpaka_common;

  // If the first argument is not a multiple of the second argument, round it up to the next multiple
  inline constexpr Idx round_up_by(Idx value, Idx divisor) { return (value + divisor - 1) / divisor * divisor; }

  // Return the integer division of the first argument by the second argument, rounded up to the next integer
  inline constexpr Idx divide_up_by(Idx value, Idx divisor) { return (value + divisor - 1) / divisor; }

  // Create an accelerator-dependent work division for 1-dimensional kernels
  template <typename TAcc,
            typename = std::enable_if_t<cms::alpakatools::is_accelerator_v<TAcc> and alpaka::Dim<TAcc>::value == 1>>
  inline WorkDiv<Dim1D> make_workdiv(Idx blocks, Idx elements) {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    if constexpr (std::is_same_v<TAcc, alpaka::AccGpuCudaRt<Dim1D, Idx>>) {
      // On GPU backends, each thread is looking at a single element:
      //   - the number of threads per block is "elements";
      //   - the number of elements per thread is always 1.
      return WorkDiv<Dim1D>(blocks, elements, Idx{1});
    } else
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED
#if ALPAKA_ACC_GPU_HIP_ENABLED
        if constexpr (std::is_same_v<TAcc, alpaka::AccGpuHipRt<Dim1D, Idx>>) {
      // On GPU backends, each thread is looking at a single element:
      //   - the number of threads per block is "elements";
      //   - the number of elements per thread is always 1.
      return WorkDiv<Dim1D>(blocks, elements, Idx{1});
    } else
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED
    {
      // On CPU backends, run serially with a single thread per block:
      //   - the number of threads per block is always 1;
      //   - the number of elements per thread is "elements".
      return WorkDiv<Dim1D>(blocks, Idx{1}, elements);
    }
  }

  // Create the accelerator-dependent workdiv for N-dimensional kernels
  template <typename TAcc, typename = std::enable_if_t<cms::alpakatools::is_accelerator_v<TAcc>>>
  inline WorkDiv<alpaka::Dim<TAcc>> make_workdiv(const Vec<alpaka::Dim<TAcc>>& blocks,
                                                 const Vec<alpaka::Dim<TAcc>>& elements) {
    using Dim = alpaka::Dim<TAcc>;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    if constexpr (std::is_same_v<TAcc, alpaka::AccGpuCudaRt<Dim, Idx>>) {
      // On GPU backends, each thread is looking at a single element:
      //   - the number of threads per block is "elements";
      //   - the number of elements per thread is always 1.
      return WorkDiv<Dim>(blocks, elements, Vec<Dim>::ones());
    } else
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
        if constexpr (std::is_same_v<TAcc, alpaka::AccGpuHipRt<Dim, Idx>>) {
      // On GPU backends, each thread is looking at a single element:
      //   - the number of threads per block is "elements";
      //   - the number of elements per thread is always 1.
      return WorkDiv<Dim>(blocks, elements, Vec<Dim>::ones());
    } else
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED
    {
      // On CPU backends, run serially with a single thread per block:
      //   - the number of threads per block is always 1;
      //   - the number of elements per thread is "elements".
      return WorkDiv<Dim>(blocks, Vec<Dim>::ones(), elements);
    }
  }

  template <typename TAcc,
            typename = std::enable_if_t<cms::alpakatools::is_accelerator_v<TAcc> and alpaka::Dim<TAcc>::value == 1>>
  class elements_with_stride {
  public:
    ALPAKA_FN_ACC inline elements_with_stride(TAcc const& acc)
        : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]},
          first_{alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u] * elements_},
          stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u] * elements_},
          extent_{stride_} {}

    ALPAKA_FN_ACC inline elements_with_stride(TAcc const& acc, Idx extent)
        : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]},
          first_{alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u] * elements_},
          stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u] * elements_},
          extent_{extent} {}

    class iterator {
      friend class elements_with_stride;

      ALPAKA_FN_ACC inline iterator(Idx elements, Idx stride, Idx extent, Idx first)
          : elements_{elements},
            stride_{stride},
            extent_{extent},
            first_{std::min(first, extent)},
            index_{first_},
            last_{std::min(first + elements, extent)} {}

    public:
      ALPAKA_FN_ACC inline Idx operator*() const { return index_; }

      // pre-increment the iterator
      ALPAKA_FN_ACC inline iterator& operator++() {
        // increment the index along the elements processed by the current thread
        ++index_;
        if (index_ < last_)
          return *this;

        // increment the thread index with the grid stride
        first_ += stride_;
        index_ = first_;
        last_ = std::min(first_ + elements_, extent_);
        if (index_ < extent_)
          return *this;

        // the iterator has reached or passed the end of the extent, clamp it to the extent
        first_ = extent_;
        index_ = extent_;
        last_ = extent_;
        return *this;
      }

      // post-increment the iterator
      ALPAKA_FN_ACC inline iterator operator++(int) {
        iterator old = *this;
        ++(*this);
        return old;
      }

      ALPAKA_FN_ACC inline bool operator==(iterator const& other) const {
        return (index_ == other.index_) and (first_ == other.first_);
      }

      ALPAKA_FN_ACC inline bool operator!=(iterator const& other) const { return not(*this == other); }

    private:
      // non-const to support iterator copy and assignment
      Idx elements_;
      Idx stride_;
      Idx extent_;
      // modified by the pre/post-increment operator
      Idx first_;
      Idx index_;
      Idx last_;
    };

    ALPAKA_FN_ACC inline iterator begin() const { return iterator(elements_, stride_, extent_, first_); }

    ALPAKA_FN_ACC inline iterator end() const { return iterator(elements_, stride_, extent_, extent_); }

  private:
    const Idx elements_;
    const Idx first_;
    const Idx stride_;
    const Idx extent_;
  };

  template <typename TAcc,
            typename = std::enable_if_t<cms::alpakatools::is_accelerator_v<TAcc> and (alpaka::Dim<TAcc>::value > 0)>>
  class elements_with_stride_nd {
  public:
    using Dim = alpaka::Dim<TAcc>;
    using Vec = alpaka::Vec<Dim, Idx>;

    ALPAKA_FN_ACC inline elements_with_stride_nd(TAcc const& acc)
        : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)},
          first_{alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc) * elements_},
          stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc) * elements_},
          extent_{stride_} {}

    ALPAKA_FN_ACC inline elements_with_stride_nd(TAcc const& acc, Vec extent)
        : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)},
          first_{alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc) * elements_},
          stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc) * elements_},
          extent_{extent} {}

    class iterator {
      friend class elements_with_stride_nd;
      constexpr static const auto last_dimension = Dim::value - 1;

      ALPAKA_FN_ACC inline iterator(Vec elements, Vec stride, Vec extent, Vec first)
          : elements_{elements},
            stride_{stride},
            extent_{extent},
            first_{alpaka::elementwise_min(first, extent)},
            index_{first_},
            last_{std::min(first[last_dimension] + elements[last_dimension], extent[last_dimension])} {}

    public:
      ALPAKA_FN_ACC inline Vec operator*() const { return index_; }

      // pre-increment the iterator
      ALPAKA_FN_ACC inline iterator& operator++() {
        // increment the index along the elements processed by the current thread
        ++index_[last_dimension];
        if (index_[last_dimension] < last_)
          return *this;

        // increment the thread index along with the last dimension with the grid stride
        first_[last_dimension] += stride_[last_dimension];
        index_[last_dimension] = first_[last_dimension];
        last_ = std::min(first_[last_dimension] + elements_[last_dimension], extent_[last_dimension]);
        if (index_[last_dimension] < extent_[last_dimension])
          return *this;

        // increment the thread index along the outer dimensions with the grid stride
        if constexpr (last_dimension > 0)
          for (auto dimension = last_dimension - 1; dimension >= 0; --dimension) {
            first_[dimension] += stride_[dimension];
            index_[dimension] = first_[dimension];
            if (index_[dimension] < extent_[dimension])
              return *this;
          }

        // the iterator has reached or passed the end of the extent, clamp it to the extent
        first_ = extent_;
        index_ = extent_;
        last_ = extent_[last_dimension];
        return *this;
      }

      // post-increment the iterator
      ALPAKA_FN_ACC inline iterator operator++(int) {
        iterator old = *this;
        ++(*this);
        return old;
      }

      ALPAKA_FN_ACC inline bool operator==(iterator const& other) const {
        return (index_ == other.index_) and (first_ == other.first_);
      }

      ALPAKA_FN_ACC inline bool operator!=(iterator const& other) const { return not(*this == other); }

    private:
      // non-const to support iterator copy and assignment
      Vec elements_;
      Vec stride_;
      Vec extent_;
      // modified by the pre/post-increment operator
      Vec first_;
      Vec index_;
      Idx last_;
    };

    ALPAKA_FN_ACC inline iterator begin() const { return iterator(elements_, stride_, extent_, first_); }

    ALPAKA_FN_ACC inline iterator end() const { return iterator(elements_, stride_, extent_, extent_); }

  private:
    const Vec elements_;
    const Vec first_;
    const Vec stride_;
    const Vec extent_;
  };

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaInterface_interface_workdivision_h
