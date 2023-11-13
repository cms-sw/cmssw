#ifndef HeterogeneousCore_AlpakaInterface_interface_workdivision_h
#define HeterogeneousCore_AlpakaInterface_interface_workdivision_h

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"

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

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
  template <typename TDim>
  struct requires_single_thread_per_block<alpaka::AccCpuThreads<TDim, Idx>> : public std::false_type {};
#endif  // ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED

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

  /* ElementIndex
   *
   * an aggregate that containes the .global and .local indices of an element; returned by iterating over elements_in_block.
   */

  struct ElementIndex {
    Idx global;
    Idx local;
  };

  /* elements_with_stride
   */

  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value == 1>>
  class elements_with_stride {
  public:
    ALPAKA_FN_ACC inline elements_with_stride(TAcc const& acc)
        : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]},
          thread_{alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u] * elements_},
          stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u] * elements_},
          extent_{stride_} {}

    ALPAKA_FN_ACC inline elements_with_stride(TAcc const& acc, Idx extent)
        : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]},
          thread_{alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u] * elements_},
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
            range_{std::min(first + elements, extent)} {}

    public:
      ALPAKA_FN_ACC inline Idx operator*() const { return index_; }

      // pre-increment the iterator
      ALPAKA_FN_ACC inline iterator& operator++() {
        if constexpr (requires_single_thread_per_block_v<TAcc>) {
          // increment the index along the elements processed by the current thread
          ++index_;
          if (index_ < range_)
            return *this;
        }

        // increment the thread index with the grid stride
        first_ += stride_;
        index_ = first_;
        range_ = std::min(first_ + elements_, extent_);
        if (index_ < extent_)
          return *this;

        // the iterator has reached or passed the end of the extent, clamp it to the extent
        first_ = extent_;
        index_ = extent_;
        range_ = extent_;
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
      Idx range_;
    };

    ALPAKA_FN_ACC inline iterator begin() const { return iterator(elements_, stride_, extent_, thread_); }

    ALPAKA_FN_ACC inline iterator end() const { return iterator(elements_, stride_, extent_, extent_); }

  private:
    const Idx elements_;
    const Idx thread_;
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
          thread_{alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc) * elements_},
          stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc) * elements_},
          extent_{stride_} {}

    ALPAKA_FN_ACC inline elements_with_stride_nd(TAcc const& acc, Vec extent)
        : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)},
          thread_{alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc) * elements_},
          stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc) * elements_},
          extent_{extent} {}

    // tag used to construct an end iterator
    struct at_end_t {};

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
      // construct an iterator pointing to the first element to be processed by the current thread
      ALPAKA_FN_ACC inline iterator(elements_with_stride_nd const* loop, Vec first)
          : loop_{loop},
            first_{alpaka::elementwise_min(first, loop->extent_)},
            range_{alpaka::elementwise_min(first + loop->elements_, loop->extent_)},
            index_{first_} {}

      // construct an end iterator, pointing post the end of the extent
      ALPAKA_FN_ACC inline iterator(elements_with_stride_nd const* loop, at_end_t const&)
          : loop_{loop}, first_{loop_->extent_}, range_{loop_->extent_}, index_{loop_->extent_} {}

      template <size_t I>
      ALPAKA_FN_ACC inline constexpr bool nth_elements_loop() {
        bool overflow = false;
        ++index_[I];
        if (index_[I] >= range_[I]) {
          index_[I] = first_[I];
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
        first_[I] += loop_->stride_[I];
        if (first_[I] >= loop_->extent_[I]) {
          first_[I] = loop_->thread_[I];
          overflow = true;
        }
        index_[I] = first_[I];
        range_[I] = std::min(first_[I] + loop_->elements_[I], loop_->extent_[I]);
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
        first_ = loop_->extent_;
        range_ = loop_->extent_;
        index_ = loop_->extent_;
      }

      // const pointer to the elements_with_stride_nd that the iterator refers to
      const elements_with_stride_nd* loop_;

      // modified by the pre/post-increment operator
      Vec first_;  // first element processed by this thread
      Vec range_;  // last element processed by this thread
      Vec index_;  // current element processed by this thread
    };

    ALPAKA_FN_ACC inline iterator begin() const {
      // check that all dimensions of the current thread index are within the extent
      if ((thread_ < extent_).all()) {
        // construct an iterator pointing to the first element to be processed by the current thread
        return iterator{this, thread_};
      } else {
        // construct an end iterator, pointing post the end of the extent
        return iterator{this, at_end_t{}};
      }
    }

    ALPAKA_FN_ACC inline iterator end() const {
      // construct an end iterator, pointing post the end of the extent
      return iterator{this, at_end_t{}};
    }

  private:
    const Vec elements_;
    const Vec thread_;
    const Vec stride_;
    const Vec extent_;
  };

  /* blocks_with_stride
   *
   * `blocks_with_stride(acc, size)` returns a range than spans the (virtual) block indices required to cover the given
   * problem size.
   *
   * For example, if size is 1000 and the block size is 16, it will return the range from 1 to 62.
   * If the work division has more than 63 blocks, only the first 63 will perform one iteration of the loop, and the
   * other will exit immediately.
   * If the work division has less than 63 blocks, some of the blocks will perform more than one iteration, in order to
   * cover then whole problem space.
   *
   * All threads in a block see the same loop iterations, while threads in different blocks may see a different number
   * of iterations.
   */

  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value == 1>>
  class blocks_with_stride {
  public:
    ALPAKA_FN_ACC inline blocks_with_stride(TAcc const& acc)
        : first_{alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]},
          stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u]},
          extent_{stride_} {}

    // extent is the total number of elements (not blocks)
    ALPAKA_FN_ACC inline blocks_with_stride(TAcc const& acc, Idx extent)
        : first_{alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]},
          stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u]},
          extent_{divide_up_by(extent, alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u])} {}

    class iterator {
      friend class blocks_with_stride;

      ALPAKA_FN_ACC inline iterator(Idx stride, Idx extent, Idx first)
          : stride_{stride}, extent_{extent}, first_{std::min(first, extent)} {}

    public:
      ALPAKA_FN_ACC inline Idx operator*() const { return first_; }

      // pre-increment the iterator
      ALPAKA_FN_ACC inline iterator& operator++() {
        // increment the first-element-in-block index by the grid stride
        first_ += stride_;
        if (first_ < extent_)
          return *this;

        // the iterator has reached or passed the end of the extent, clamp it to the extent
        first_ = extent_;
        return *this;
      }

      // post-increment the iterator
      ALPAKA_FN_ACC inline iterator operator++(int) {
        iterator old = *this;
        ++(*this);
        return old;
      }

      ALPAKA_FN_ACC inline bool operator==(iterator const& other) const { return (first_ == other.first_); }

      ALPAKA_FN_ACC inline bool operator!=(iterator const& other) const { return not(*this == other); }

    private:
      // non-const to support iterator copy and assignment
      Idx stride_;
      Idx extent_;
      // modified by the pre/post-increment operator
      Idx first_;
    };

    ALPAKA_FN_ACC inline iterator begin() const { return iterator(stride_, extent_, first_); }

    ALPAKA_FN_ACC inline iterator end() const { return iterator(stride_, extent_, extent_); }

  private:
    const Idx first_;
    const Idx stride_;
    const Idx extent_;
  };

  /* elements_in_block
   *
   * `elements_in_block(acc, block, size)` returns a range that spans all the elements within the given block.
   * Iterating over the range yields values of type ElementIndex, that contain both .global and .local indices
   * of the corresponding element.
   *
   * If the work division has only one element per thread, the loop will perform at most one iteration.
   * If the work division has more than one elements per thread, the loop will perform that number of iterations,
   * or less if it reaches size.
   */

  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value == 1>>
  class elements_in_block {
  public:
    ALPAKA_FN_ACC inline elements_in_block(TAcc const& acc, Idx block)
        : first_{block * alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]},
          local_{alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] *
                 alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]},
          range_{local_ + alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]} {}

    ALPAKA_FN_ACC inline elements_in_block(TAcc const& acc, Idx block, Idx extent)
        : first_{block * alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]},
          local_{std::min(extent - first_,
                          alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] *
                              alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u])},
          range_{std::min(extent - first_, local_ + alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u])} {}

    class iterator {
      friend class elements_in_block;

      ALPAKA_FN_ACC inline iterator(Idx local, Idx first, Idx range) : index_{local}, first_{first}, range_{range} {}

    public:
      ALPAKA_FN_ACC inline ElementIndex operator*() const { return ElementIndex{index_ + first_, index_}; }

      // pre-increment the iterator
      ALPAKA_FN_ACC inline iterator& operator++() {
        if constexpr (requires_single_thread_per_block_v<TAcc>) {
          // increment the index along the elements processed by the current thread
          ++index_;
          if (index_ < range_)
            return *this;
        }

        // the iterator has reached or passed the end of the extent, clamp it to the extent
        index_ = range_;
        return *this;
      }

      // post-increment the iterator
      ALPAKA_FN_ACC inline iterator operator++(int) {
        iterator old = *this;
        ++(*this);
        return old;
      }

      ALPAKA_FN_ACC inline bool operator==(iterator const& other) const { return (index_ == other.index_); }

      ALPAKA_FN_ACC inline bool operator!=(iterator const& other) const { return not(*this == other); }

    private:
      // modified by the pre/post-increment operator
      Idx index_;
      // non-const to support iterator copy and assignment
      Idx first_;
      Idx range_;
    };

    ALPAKA_FN_ACC inline iterator begin() const { return iterator(local_, first_, range_); }

    ALPAKA_FN_ACC inline iterator end() const { return iterator(range_, first_, range_); }

  private:
    const Idx first_;
    const Idx local_;
    const Idx range_;
  };

  /* once_per_grid
   *
   * `once_per_grid(acc)` returns true for a single thread within the kernel execution grid.
   *
   * Usually the condition is true for block 0 and thread 0, but these indices should not be relied upon.
   */

  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC inline constexpr bool once_per_grid(TAcc const& acc) {
    return alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc) == Vec<alpaka::Dim<TAcc>>::zeros();
  }

  /* once_per_block
   *
   * `once_per_block(acc)` returns true for a single thread within the block.
   *
   * Usually the condition is true for thread 0, but this index should not be relied upon.
   */

  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC inline constexpr bool once_per_block(TAcc const& acc) {
    return alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc) == Vec<alpaka::Dim<TAcc>>::zeros();
  }

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaInterface_interface_workdivision_h
