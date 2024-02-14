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
   *
   * `elements_with_stride(acc, [first, ]extent)` returns an iteratable range that spans the element indices required to
   * cover the given problem size:
   *   - `first` (optional) is index to the first element; if not specified, the loop starts from 0;
   *   - `extent` is the total size of the problem, including any elements that may come before `first`.
   *
   * To cover the problem space, different threads may execute a different number of iterations. As a result, it is not
   * safe to call alpaka::syncBlockThreads() within this loop. If a block synchronisation is needed, one should split
   * the loop into an outer loop on the blocks and an inner loop on the threads, and call the syncronisation only in the
   * outer loop:
   *
   *  for (auto group : uniform_groups(acc, extent) {
   *    for (auto element : uniform_group_elements(acc, group, extent) {
   *       // no synchronisations here
   *       ...
   *    }
   *    alpaka::syncBlockThreads();
   *    for (auto element : uniform_group_elements(acc, group, extent) {
   *       // no synchronisations here
   *       ...
   *    }
   *    alpaka::syncBlockThreads();
   *  }
   */

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

    ALPAKA_FN_ACC inline elements_with_stride(TAcc const& acc, Idx first, Idx extent)
        : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]},
          first_{alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u] * elements_ + first},
          stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u] * elements_},
          extent_{extent} {}

    class const_iterator;
    using iterator = const_iterator;

    ALPAKA_FN_ACC inline const_iterator begin() const { return const_iterator(elements_, stride_, extent_, first_); }

    ALPAKA_FN_ACC inline const_iterator end() const { return const_iterator(elements_, stride_, extent_, extent_); }

    class const_iterator {
      friend class elements_with_stride;

      ALPAKA_FN_ACC inline const_iterator(Idx elements, Idx stride, Idx extent, Idx first)
          : elements_{elements},
            stride_{stride},
            extent_{extent},
            first_{std::min(first, extent)},
            index_{first_},
            range_{std::min(first + elements, extent)} {}

    public:
      ALPAKA_FN_ACC inline Idx operator*() const { return index_; }

      // pre-increment the iterator
      ALPAKA_FN_ACC inline const_iterator& operator++() {
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
      ALPAKA_FN_ACC inline const_iterator operator++(int) {
        const_iterator old = *this;
        ++(*this);
        return old;
      }

      ALPAKA_FN_ACC inline bool operator==(const_iterator const& other) const {
        return (index_ == other.index_) and (first_ == other.first_);
      }

      ALPAKA_FN_ACC inline bool operator!=(const_iterator const& other) const { return not(*this == other); }

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

    class const_iterator;
    using iterator = const_iterator;

    ALPAKA_FN_ACC inline const_iterator begin() const {
      // check that all dimensions of the current thread index are within the extent
      if ((thread_ < extent_).all()) {
        // construct an iterator pointing to the first element to be processed by the current thread
        return const_iterator{this, thread_};
      } else {
        // construct an end iterator, pointing post the end of the extent
        return const_iterator{this, at_end_t{}};
      }
    }

    ALPAKA_FN_ACC inline const_iterator end() const {
      // construct an end iterator, pointing post the end of the extent
      return const_iterator{this, at_end_t{}};
    }

    class const_iterator {
      friend class elements_with_stride_nd;

    public:
      ALPAKA_FN_ACC inline Vec operator*() const { return index_; }

      // pre-increment the iterator
      ALPAKA_FN_ACC constexpr inline const_iterator operator++() {
        increment();
        return *this;
      }

      // post-increment the iterator
      ALPAKA_FN_ACC constexpr inline const_iterator operator++(int) {
        const_iterator old = *this;
        increment();
        return old;
      }

      ALPAKA_FN_ACC constexpr inline bool operator==(const_iterator const& other) const {
        return (index_ == other.index_);
      }

      ALPAKA_FN_ACC constexpr inline bool operator!=(const_iterator const& other) const { return not(*this == other); }

    private:
      // construct an iterator pointing to the first element to be processed by the current thread
      ALPAKA_FN_ACC inline const_iterator(elements_with_stride_nd const* loop, Vec first)
          : loop_{loop},
            first_{alpaka::elementwise_min(first, loop->extent_)},
            range_{alpaka::elementwise_min(first + loop->elements_, loop->extent_)},
            index_{first_} {}

      // construct an end iterator, pointing post the end of the extent
      ALPAKA_FN_ACC inline const_iterator(elements_with_stride_nd const* loop, at_end_t const&)
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

    class const_iterator;
    using iterator = const_iterator;

    ALPAKA_FN_ACC inline const_iterator begin() const { return const_iterator(stride_, extent_, first_); }

    ALPAKA_FN_ACC inline const_iterator end() const { return const_iterator(stride_, extent_, extent_); }

    class const_iterator {
      friend class blocks_with_stride;

      ALPAKA_FN_ACC inline const_iterator(Idx stride, Idx extent, Idx first)
          : stride_{stride}, extent_{extent}, first_{std::min(first, extent)} {}

    public:
      ALPAKA_FN_ACC inline Idx operator*() const { return first_; }

      // pre-increment the iterator
      ALPAKA_FN_ACC inline const_iterator& operator++() {
        // increment the first-element-in-block index by the grid stride
        first_ += stride_;
        if (first_ < extent_)
          return *this;

        // the iterator has reached or passed the end of the extent, clamp it to the extent
        first_ = extent_;
        return *this;
      }

      // post-increment the iterator
      ALPAKA_FN_ACC inline const_iterator operator++(int) {
        const_iterator old = *this;
        ++(*this);
        return old;
      }

      ALPAKA_FN_ACC inline bool operator==(const_iterator const& other) const { return (first_ == other.first_); }

      ALPAKA_FN_ACC inline bool operator!=(const_iterator const& other) const { return not(*this == other); }

    private:
      // non-const to support iterator copy and assignment
      Idx stride_;
      Idx extent_;
      // modified by the pre/post-increment operator
      Idx first_;
    };

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
   *
   * If the problem size is not a multiple of the block size, different threads may execute a different number of
   * iterations. As a result, it is not safe to call alpaka::syncBlockThreads() within this loop. If a block
   * synchronisation is needed, one should split the loop, and synchronise the threads between the loops.
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

    class const_iterator;
    using iterator = const_iterator;

    ALPAKA_FN_ACC inline const_iterator begin() const { return const_iterator(local_, first_, range_); }

    ALPAKA_FN_ACC inline const_iterator end() const { return const_iterator(range_, first_, range_); }

    class const_iterator {
      friend class elements_in_block;

      ALPAKA_FN_ACC inline const_iterator(Idx local, Idx first, Idx range)
          : index_{local}, first_{first}, range_{range} {}

    public:
      ALPAKA_FN_ACC inline ElementIndex operator*() const { return ElementIndex{index_ + first_, index_}; }

      // pre-increment the iterator
      ALPAKA_FN_ACC inline const_iterator& operator++() {
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
      ALPAKA_FN_ACC inline const_iterator operator++(int) {
        const_iterator old = *this;
        ++(*this);
        return old;
      }

      ALPAKA_FN_ACC inline bool operator==(const_iterator const& other) const { return (index_ == other.index_); }

      ALPAKA_FN_ACC inline bool operator!=(const_iterator const& other) const { return not(*this == other); }

    private:
      // modified by the pre/post-increment operator
      Idx index_;
      // non-const to support iterator copy and assignment
      Idx first_;
      Idx range_;
    };

  private:
    const Idx first_;
    const Idx local_;
    const Idx range_;
  };

  /* uniform_groups
   *
   * `uniform_groups(acc, elements)` returns a range than spans the group indices required to cover the given problem
   * size, in units of the block size:
   *   - the `elements` argument indicates the total number of elements, across all groups.
   *
   * `uniform_groups` should be called consistently by all the threads in a block. All threads in a block see the same
   * loop iterations, while threads in different blocks may see a different number of iterations.
   *
   * For example, if `size` is 1000 and the block size is 16,
   *
   *   for (auto group: uniform_groups(acc, 1000)
   *
   * will return the range from 0 to 62, split across all blocks in the work division.
   *
   * If the work division has more than 63 blocks, the first 63 will perform one iteration of the loop, while the other
   * blocks will exit immediately.
   * If the work division has less than 63 blocks, some of the blocks will perform more than one iteration, in order to
   * cover then whole problem space.
   *
   * If the problem size is not a multiple of the block size, the last group will process a number of elements smaller
   * than the block size. Also in this case all threads in the block will execute the same number of iterations of this
   * loop: this makes it safe to use block-level synchronisations in the loop body. It is left to the inner loop (or the
   * user) to ensure that only the correct number of threads process any data.
   */

  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value == 1>>
  using uniform_groups = blocks_with_stride<TAcc>;

  /* uniform_group_elements
   *
   * `uniform_group_elements(acc, group, elements)` returns a range that spans all the elements within the given group:
   *   - the `group` argument indicates the id of the current group, for example as obtained from `uniform_groups`;
   *   - the `elements` argument indicates the total number of elements, across all groups.
   *
   * Iterating over the range yields values of type `ElementIndex`, that contain the `.global` and `.local` indices of
   * the corresponding element.
   *
   * The loop will perform a number of iterations up to the number of elements per thread, stopping earlier when the
   * element index reaches `size`.
   *
   * If the problem size is not a multiple of the block size, different threads may execute a different number of
   * iterations. As a result, it is not safe to call alpaka::syncBlockThreads() within this loop. If a block
   * synchronisation is needed, one should split the loop, and synchronise the threads between the loops.
   */

  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value == 1>>
  using uniform_group_elements = elements_in_block<TAcc>;

  /* independent_groups
   *
   * `independent_groups(acc, groups)` returns a range than spans the group indices from 0 to `groups`, with one group
   * per block:
   *   - the `groups` argument indicates the total number of groups.
   *
   * If the work division has more blocks than `groups`, only the first `groups` blocks will perform one iteration of
   * the loop, while the other blocks will exit immediately.
   * If the work division has less blocks than `groups`, some of the blocks will perform more than one iteration, in
   * order to cover then whole problem space.
   */

  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value == 1>>
  class independent_groups {
  public:
    ALPAKA_FN_ACC inline independent_groups(TAcc const& acc)
        : first_{alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]},
          stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u]},
          extent_{stride_} {}

    // extent is the total number of elements (not blocks)
    ALPAKA_FN_ACC inline independent_groups(TAcc const& acc, Idx groups)
        : first_{alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]},
          stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u]},
          extent_{groups} {}

    class const_iterator;
    using iterator = const_iterator;

    ALPAKA_FN_ACC inline const_iterator begin() const { return const_iterator(stride_, extent_, first_); }

    ALPAKA_FN_ACC inline const_iterator end() const { return const_iterator(stride_, extent_, extent_); }

    class const_iterator {
      friend class independent_groups;

      ALPAKA_FN_ACC inline const_iterator(Idx stride, Idx extent, Idx first)
          : stride_{stride}, extent_{extent}, first_{std::min(first, extent)} {}

    public:
      ALPAKA_FN_ACC inline Idx operator*() const { return first_; }

      // pre-increment the iterator
      ALPAKA_FN_ACC inline const_iterator& operator++() {
        // increment the first-element-in-block index by the grid stride
        first_ += stride_;
        if (first_ < extent_)
          return *this;

        // the iterator has reached or passed the end of the extent, clamp it to the extent
        first_ = extent_;
        return *this;
      }

      // post-increment the iterator
      ALPAKA_FN_ACC inline const_iterator operator++(int) {
        const_iterator old = *this;
        ++(*this);
        return old;
      }

      ALPAKA_FN_ACC inline bool operator==(const_iterator const& other) const { return (first_ == other.first_); }

      ALPAKA_FN_ACC inline bool operator!=(const_iterator const& other) const { return not(*this == other); }

    private:
      // non-const to support iterator copy and assignment
      Idx stride_;
      Idx extent_;
      // modified by the pre/post-increment operator
      Idx first_;
    };

  private:
    const Idx first_;
    const Idx stride_;
    const Idx extent_;
  };

  /* independent_group_elements
   *
   * `independent_group_elements(acc, elements)` returns a range that spans all the elements within the given group:
   *   - the `elements` argument indicates the number of elements in the current group.
   *
   * Iterating over the range yields the local element index, between `0` and `elements - 1`. The threads in the block
   * will perform one or more iterations, depending on the number of elements per thread, and on the number of threads
   * per block, compared with the total number of elements.
   *
   * If the problem size is not a multiple of the block size, different threads may execute a different number of
   * iterations. As a result, it is not safe to call alpaka::syncBlockThreads() within this loop. If a block
   * synchronisation is needed, one should split the loop, and synchronise the threads between the loops.
   */

  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value == 1>>
  class independent_group_elements {
  public:
    ALPAKA_FN_ACC inline independent_group_elements(TAcc const& acc)
        : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]},
          thread_{alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] * elements_},
          stride_{alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u] * elements_},
          extent_{stride_} {}

    ALPAKA_FN_ACC inline independent_group_elements(TAcc const& acc, Idx extent)
        : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]},
          thread_{alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] * elements_},
          stride_{alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u] * elements_},
          extent_{extent} {}

    ALPAKA_FN_ACC inline independent_group_elements(TAcc const& acc, Idx first, Idx extent)
        : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]},
          thread_{alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] * elements_ + first},
          stride_{alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u] * elements_},
          extent_{extent} {}

    class const_iterator;
    using iterator = const_iterator;

    ALPAKA_FN_ACC inline const_iterator begin() const { return const_iterator(elements_, stride_, extent_, thread_); }

    ALPAKA_FN_ACC inline const_iterator end() const { return const_iterator(elements_, stride_, extent_, extent_); }

    class const_iterator {
      friend class independent_group_elements;

      ALPAKA_FN_ACC inline const_iterator(Idx elements, Idx stride, Idx extent, Idx first)
          : elements_{elements},
            stride_{stride},
            extent_{extent},
            first_{std::min(first, extent)},
            index_{first_},
            range_{std::min(first + elements, extent)} {}

    public:
      ALPAKA_FN_ACC inline Idx operator*() const { return index_; }

      // pre-increment the iterator
      ALPAKA_FN_ACC inline const_iterator& operator++() {
        if constexpr (requires_single_thread_per_block_v<TAcc>) {
          // increment the index along the elements processed by the current thread
          ++index_;
          if (index_ < range_)
            return *this;
        }

        // increment the thread index with the block stride
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
      ALPAKA_FN_ACC inline const_iterator operator++(int) {
        const_iterator old = *this;
        ++(*this);
        return old;
      }

      ALPAKA_FN_ACC inline bool operator==(const_iterator const& other) const {
        return (index_ == other.index_) and (first_ == other.first_);
      }

      ALPAKA_FN_ACC inline bool operator!=(const_iterator const& other) const { return not(*this == other); }

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

  private:
    const Idx elements_;
    const Idx thread_;
    const Idx stride_;
    const Idx extent_;
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
