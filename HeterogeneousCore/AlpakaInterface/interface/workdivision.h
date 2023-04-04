#ifndef HeterogeneousCore_AlpakaInterface_interface_workdivision_h
#define HeterogeneousCore_AlpakaInterface_interface_workdivision_h

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
// #include "HeterogeneousCore/AlpakaInterface/interface/vec.h"

namespace cms::alpakatools {

  using namespace alpaka_common;

  // If the first argument is not a multiple of the second argument, round it up to the next multiple
  inline constexpr Idx round_up_by(Idx value, Idx divisor) { return (value + divisor - 1) / divisor * divisor; }

  // Return the integer division of the first argument by the second argument, rounded up to the next integer
  inline constexpr Idx divide_up_by(Idx value, Idx divisor) { return (value + divisor - 1) / divisor; }

  // Trait describing whether or not the accelerator expects the threads-per-block and elements-per-thread to be swapped
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  struct requires_single_thread_per_block : public std::true_type {};

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  template <typename TDim>
  struct requires_single_thread_per_block<alpaka::AccGpuCudaRt<TDim, Idx>> : public std::false_type {};
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
  template <typename TDim>
  struct requires_single_thread_per_block<alpaka::AccGpuHipRt<TDim, Idx>> : public std::false_type {};
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

  // Whether or not the accelerator expects the threads-per-block and elements-per-thread to be swapped
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  inline constexpr bool requires_single_thread_per_block_v = requires_single_thread_per_block<TAcc>::value;

  // Create an accelerator-dependent work division for 1-dimensional kernels
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value == 1>>
  inline WorkDiv<Dim1D> make_workdiv(Idx blocks, Idx elements) {
    if constexpr (not requires_single_thread_per_block_v<TAcc>) {
      // On GPU backends, each thread is looking at a single element:
      //   - the number of threads per block is "elements";
      //   - the number of elements per thread is always 1.
      return WorkDiv<Dim1D>(blocks, elements, Idx{1});
    } else {
      // On CPU backends, run serially with a single thread per block:
      //   - the number of threads per block is always 1;
      //   - the number of elements per thread is "elements".
      return WorkDiv<Dim1D>(blocks, Idx{1}, elements);
    }
  }

  // Create the accelerator-dependent workdiv for N-dimensional kernels
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  inline WorkDiv<alpaka::Dim<TAcc>> make_workdiv(const Vec<alpaka::Dim<TAcc>>& blocks,
                                                 const Vec<alpaka::Dim<TAcc>>& elements) {
    using Dim = alpaka::Dim<TAcc>;
    if constexpr (not requires_single_thread_per_block_v<TAcc>) {
      // On GPU backends, each thread is looking at a single element:
      //   - the number of threads per block is "elements";
      //   - the number of elements per thread is always 1.
      return WorkDiv<Dim>(blocks, elements, Vec<Dim>::ones());
    } else {
      // On CPU backends, run serially with a single thread per block:
      //   - the number of threads per block is always 1;
      //   - the number of elements per thread is "elements".
      return WorkDiv<Dim>(blocks, Vec<Dim>::ones(), elements);
    }
  }

  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value == 1>>
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
        if constexpr (requires_single_thread_per_block_v<TAcc>) {
          // increment the index along the elements processed by the current thread
          ++index_;
          if (index_ < last_)
            return *this;
        }

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

  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 0)>>
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

    public:
      ALPAKA_FN_ACC inline Vec operator*() const { return index_; }

      // pre-increment the iterator
      ALPAKA_FN_ACC constexpr inline iterator operator++() {
        increment();
        return *this;
      }

      // post-increment the iterator
      ALPAKA_FN_ACC constexpr inline iterator operator++(int) {
        iterator old = *this;
        increment();
        return old;
      }

      ALPAKA_FN_ACC constexpr inline bool operator==(iterator const& other) const { return (index_ == other.index_); }

      ALPAKA_FN_ACC constexpr inline bool operator!=(iterator const& other) const { return not(*this == other); }

    private:
      // private, explicit constructor
      ALPAKA_FN_ACC inline iterator(elements_with_stride_nd const* loop, Vec first)
          : loop_{loop},
            thread_{alpaka::elementwise_min(first, loop->extent_)},
            range_{alpaka::elementwise_min(first + loop->elements_, loop->extent_)},
            index_{thread_} {}

      template <size_t I>
      ALPAKA_FN_ACC inline constexpr bool nth_elements_loop() {
        bool overflow = false;
        ++index_[I];
        if (index_[I] >= range_[I]) {
          index_[I] = thread_[I];
          overflow = true;
        }
        return overflow;
      }

      template <size_t N>
      ALPAKA_FN_ACC inline constexpr bool do_elements_loops() {
        if constexpr (N == 0) {
          // overflow
          return true;
        } else {
          if (not nth_elements_loop<N - 1>()) {
            return false;
          } else {
            return do_elements_loops<N - 1>();
          }
        }
      }

      template <size_t I>
      ALPAKA_FN_ACC inline constexpr bool nth_strided_loop() {
        bool overflow = false;
        thread_[I] += loop_->stride_[I];
        if (thread_[I] >= loop_->extent_[I]) {
          thread_[I] = loop_->first_[I];
          overflow = true;
        }
        index_[I] = thread_[I];
        range_[I] = std::min(thread_[I] + loop_->elements_[I], loop_->extent_[I]);
        return overflow;
      }

      template <size_t N>
      ALPAKA_FN_ACC inline constexpr bool do_strided_loops() {
        if constexpr (N == 0) {
          // overflow
          return true;
        } else {
          if (not nth_strided_loop<N - 1>()) {
            return false;
          } else {
            return do_strided_loops<N - 1>();
          }
        }
      }

      // increment the iterator
      ALPAKA_FN_ACC inline constexpr void increment() {
        if constexpr (requires_single_thread_per_block_v<TAcc>) {
          // linear N-dimensional loops over the elements associated to the thread;
          // do_elements_loops<>() returns true if any of those loops overflows
          if (not do_elements_loops<Dim::value>()) {
            // the elements loops did not overflow, return the next index
            return;
          }
        }

        // strided N-dimensional loop over the threads in the kernel launch grid;
        // do_strided_loops<>() returns true if any of those loops overflows
        if (not do_strided_loops<Dim::value>()) {
          // the strided loops did not overflow, return the next index
          return;
        }

        // the iterator has reached or passed the end of the extent, clamp it to the extent
        thread_ = loop_->extent_;
        range_ = loop_->extent_;
        index_ = loop_->extent_;
      }

      // const pointer to the elements_with_stride_nd that the iterator refers to
      const elements_with_stride_nd* loop_;

      // modified by the pre/post-increment operator
      Vec thread_;  // first element processed by this thread
      Vec range_;   // last element processed by this thread
      Vec index_;   // current element processed by this thread
    };

    ALPAKA_FN_ACC inline iterator begin() const { return iterator{this, first_}; }

    ALPAKA_FN_ACC inline iterator end() const { return iterator{this, extent_}; }

  private:
    const Vec elements_;
    const Vec first_;
    const Vec stride_;
    const Vec extent_;
  };

  /*
   * Computes the range of the elements indexes, local to the block.
   * Truncated by the max number of elements of interest.
   */
  // template <typename TAcc>
  // ALPAKA_FN_ACC std::pair<Idx, Idx> element_index_range_in_block_truncated(const TAcc& acc,
  //                                                                          const Idx maxNumberOfElements,
  //                                                                          const Idx elementIdxShift,
  //                                                                          const unsigned int dimIndex = 0u) {
  //   // Check dimension
  //   //static_assert(alpaka::Dim<TAcc>::value == Dim1D::value,
  //   //              "Accelerator and maxNumberOfElements need to have same dimension.");
  //   auto [firstElementIdxLocal, endElementIdxLocal] = element_index_range_in_block(acc, elementIdxShift, dimIndex);

  //   // Truncate
  //   endElementIdxLocal = std::min(endElementIdxLocal, maxNumberOfElements);

  //   // Return element indexes, shifted by elementIdxShift, and truncated by maxNumberOfElements.
  //   return {firstElementIdxLocal, endElementIdxLocal};
  // }

  /*********************************************
     *           RANGE COMPUTATION
     ********************************************/

  /*
     * Computes the range of the elements indexes, local to the block.
     * Warning: the max index is not truncated by the max number of elements of interest.
     */
  template <typename TAcc>
  ALPAKA_FN_ACC std::pair<Idx, Idx> element_index_range_in_block(const TAcc& acc,
                                                                 const Idx elementIdxShift,
                                                                 const unsigned int dimIndex = 0u) {
    // Take into account the thread index in block.
    const Idx threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[dimIndex]);
    const Idx threadDimension(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[dimIndex]);

    // Compute the elements indexes in block.
    // Obviously relevant for CPU only.
    // For GPU, threadDimension == 1, and elementIdx == firstElementIdx == threadIdx + elementIdxShift.
    const Idx firstElementIdxLocal = threadIdxLocal * threadDimension;
    const Idx firstElementIdx = firstElementIdxLocal + elementIdxShift;  // Add the shift!
    const Idx endElementIdxUncut = firstElementIdx + threadDimension;

    // Return element indexes, shifted by elementIdxShift.
    return {firstElementIdx, endElementIdxUncut};
  }

  /*
     * Computes the range of the elements indexes, local to the block.
     * Truncated by the max number of elements of interest.
     */
  template <typename TAcc>
  ALPAKA_FN_ACC std::pair<Idx, Idx> element_index_range_in_block_truncated(const TAcc& acc,
                                                                           const Idx maxNumberOfElements,
                                                                           const Idx elementIdxShift,
                                                                           const unsigned int dimIndex = 0u) {
    // Check dimension
    //static_assert(alpaka::Dim<TAcc>::value == Dim1::value,
    //              "Accelerator and maxNumberOfElements need to have same dimension.");
    auto [firstElementIdxLocal, endElementIdxLocal] = element_index_range_in_block(acc, elementIdxShift, dimIndex);

    // Truncate
    endElementIdxLocal = std::min(endElementIdxLocal, maxNumberOfElements);

    // Return element indexes, shifted by elementIdxShift, and truncated by maxNumberOfElements.
    return {firstElementIdxLocal, endElementIdxLocal};
  }

  /*
     * Computes the range of the elements indexes in grid.
     * Warning: the max index is not truncated by the max number of elements of interest.
     */
  template <typename TAcc>
  ALPAKA_FN_ACC std::pair<Idx, Idx> element_index_range_in_grid(const TAcc& acc,
                                                                Idx elementIdxShift,
                                                                const unsigned int dimIndex = 0u) {
    // Take into account the block index in grid.
    const Idx blockIdxInGrid(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[dimIndex]);
    const Idx blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[dimIndex]);

    // Shift to get global indices in grid (instead of local to the block)
    elementIdxShift += blockIdxInGrid * blockDimension;

    // Return element indexes, shifted by elementIdxShift.
    return element_index_range_in_block(acc, elementIdxShift, dimIndex);
  }

  /*
   * Loop on all (CPU) elements.
   * Elements loop makes sense in CPU case only. In GPU case, elementIdx = firstElementIdx = threadIdx + shift.
   * Indexes are local to the BLOCK.
   */
  template <typename TAcc, typename Func>
  ALPAKA_FN_ACC void for_each_element_in_block(const TAcc& acc,
                                               const Idx maxNumberOfElements,
                                               const Idx elementIdxShift,
                                               const Func func,
                                               const unsigned int dimIndex = 0) {
    const auto& [firstElementIdx, endElementIdx] =
        element_index_range_in_block_truncated(acc, maxNumberOfElements, elementIdxShift, dimIndex);

    for (Idx elementIdx = firstElementIdx; elementIdx < endElementIdx; ++elementIdx) {
      func(elementIdx);
    }
  }

  /*
   * Overload for elementIdxShift = 0
   */
  template <typename TAcc, typename Func>
  ALPAKA_FN_ACC void for_each_element_in_block(const TAcc& acc,
                                               const Idx maxNumberOfElements,
                                               const Func func,
                                               const unsigned int dimIndex = 0) {
    const Idx elementIdxShift = 0;
    for_each_element_in_block(acc, maxNumberOfElements, elementIdxShift, func, dimIndex);
  }

  /**************************************************************
     *          LOOP ON ALL ELEMENTS WITH ONE LOOP
     **************************************************************/

  /*
     * Case where the input index i has reached the end of threadDimension: strides the input index.
     * Otherwise: do nothing.
     * NB 1: This helper function is used as a trick to only have one loop (like in legacy), instead of 2 loops
     * (like in all the other Alpaka helpers, 'for_each_element_in_block_strided' for example, 
     * because of the additional loop over elements in Alpaka model). 
     * This allows to keep the 'continue' and 'break' statements as-is from legacy code, 
     * and hence avoids a lot of legacy code reshuffling.
     * NB 2: Modifies i, firstElementIdx and endElementIdx.
     */
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool next_valid_element_index_strided(
      Idx& i, Idx& firstElementIdx, Idx& endElementIdx, const Idx stride, const Idx maxNumberOfElements) {
    bool isNextStrideElementValid = true;
    if (i == endElementIdx) {
      firstElementIdx += stride;
      endElementIdx += stride;
      i = firstElementIdx;
      if (i >= maxNumberOfElements) {
        isNextStrideElementValid = false;
      }
    }
    return isNextStrideElementValid;
  }

  template <typename TAcc, typename Func>
  ALPAKA_FN_ACC void for_each_element_in_block_strided(const TAcc& acc,
                                                       const Idx maxNumberOfElements,
                                                       const Idx elementIdxShift,
                                                       const Func func,
                                                       const unsigned int dimIndex = 0) {
    // Get thread / element indices in block.
    const auto& [firstElementIdxNoStride, endElementIdxNoStride] =
        element_index_range_in_block(acc, elementIdxShift, dimIndex);

    // Stride = block size.
    const Idx blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[dimIndex]);

    // Strided access.
    for (Idx threadIdx = firstElementIdxNoStride, endElementIdx = endElementIdxNoStride;
         threadIdx < maxNumberOfElements;
         threadIdx += blockDimension, endElementIdx += blockDimension) {
      // (CPU) Loop on all elements.
      if (endElementIdx > maxNumberOfElements) {
        endElementIdx = maxNumberOfElements;
      }
      for (Idx i = threadIdx; i < endElementIdx; ++i) {
        func(i);
      }
    }
  }

  /*
   * Overload for elementIdxShift = 0
   */
  template <typename TAcc, typename Func>
  ALPAKA_FN_ACC void for_each_element_in_block_strided(const TAcc& acc,
                                                       const Idx maxNumberOfElements,
                                                       const Func func,
                                                       const unsigned int dimIndex = 0) {
    const Idx elementIdxShift = 0;
    for_each_element_in_block_strided(acc, maxNumberOfElements, elementIdxShift, func, dimIndex);
  }

  template <typename TAcc, typename Func>
  ALPAKA_FN_ACC void for_each_element_in_grid_strided(const TAcc& acc,
                                                      const Idx maxNumberOfElements,
                                                      const Idx elementIdxShift,
                                                      const Func func,
                                                      const unsigned int dimIndex = 0) {
    // Get thread / element indices in block.
    const auto& [firstElementIdxNoStride, endElementIdxNoStride] =
        element_index_range_in_grid(acc, elementIdxShift, dimIndex);

    // Stride = grid size.
    const Idx gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Elems>(acc)[dimIndex]);

    // Strided access.
    for (Idx threadIdx = firstElementIdxNoStride, endElementIdx = endElementIdxNoStride;
         threadIdx < maxNumberOfElements;
         threadIdx += gridDimension, endElementIdx += gridDimension) {
      // (CPU) Loop on all elements.
      if (endElementIdx > maxNumberOfElements) {
        endElementIdx = maxNumberOfElements;
      }
      for (Idx i = threadIdx; i < endElementIdx; ++i) {
        func(i);
      }
    }
  }

  /*
   * Overload for elementIdxShift = 0
   */
  template <typename TAcc, typename Func>
  ALPAKA_FN_ACC void for_each_element_in_grid_strided(const TAcc& acc,
                                                      const Idx maxNumberOfElements,
                                                      const Func func,
                                                      const unsigned int dimIndex = 0) {
    const Idx elementIdxShift = 0;
    for_each_element_in_grid_strided(acc, maxNumberOfElements, elementIdxShift, func, dimIndex);
  }

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaInterface_interface_workdivision_h
