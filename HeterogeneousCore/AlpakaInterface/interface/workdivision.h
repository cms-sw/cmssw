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
   * an aggregate that containes the `.global` and `.local` indices of an element; returned by iterating over the objecs
   * returned by `elements_in_block` and similar functions.
   */

  struct ElementIndex {
    Idx global;
    Idx local;
  };

  /* uniform_elements_along
   *
   * `uniform_elements_along<Dim>(acc [, first], extent)` returns a one-dimensional iteratable range that spans the
   * element indices from `first` (inclusive) to `extent` (exlusive) along the `Dim` dimension.
   * If `first` is not specified, it defaults to 0.
   * If `extent` is not specified, it defaults to the kernel grid size along the `Dim` dimension.
   *
   * In a 1-dimensional kernel, `uniform_elements(acc, ...)` is a shorthand for `uniform_elements_along<0>(acc, ...)`.
   *
   * In an N-dimensional kernel, dimension 0 is the one that increases more slowly (e.g. the outer loop), followed
   * by dimension 1, up to dimension N-1 that increases fastest (e.g. the inner loop).
   * For convenience when converting CUDA or HIP code, `uniform_elements_x(acc, ...)`, `_y` and `_z` are shorthands for 
   * `uniform_elements_along<N-1>(acc, ...)`, `<N-2>` and `<N-3>`.
   *
   * To cover the problem space, different threads may execute a different number of iterations. As a result, it is not
   * safe to call `alpaka::syncBlockThreads()` and other block-level synchronisations within this loop.
   * If a block synchronisation is needed, one should split the loop into an outer loop over the groups and an inner
   * loop over each group's elements, and synchronise only in the outer loop:
   *
   *  for (auto group : uniform_groups_along<Dim>(acc, extent)) {
   *    for (auto element : uniform_group_elements_along<Dim>(acc, group, extent)) {
   *       // first part of the computation
   *       // no synchronisations here
   *       ...
   *    }
   *    // wait for all threads to complete the first part
   *    alpaka::syncBlockThreads();
   *    for (auto element : uniform_group_elements_along<Dim>(acc, group, extent)) {
   *       // second part of the computation
   *       // no synchronisations here
   *       ...
   *    }
   *    // wait for all threads to complete the second part
   *    alpaka::syncBlockThreads();
   *    ...
   *  }
   *
   * Warp-level primitives require that all threads in the warp execute the same function. If `extent` is not a multiple
   * of the warp size, some of the warps may be incomplete, leading to undefined behaviour - for example, the kernel may
   * hang. To avoid this problem, round up `extent` to a multiple of the warp size, and check the element index
   * explicitly inside the loop:
   *
   *  for (auto element : uniform_elements_along<N-1>(acc, round_up_by(extent, alpaka::warp::getSize(acc)))) {
   *    bool flag = false;
   *    if (element < extent) {
   *      // do some work and compute a result flag only for the valid elements
   *      flag = do_some_work();
   *    }
   *    // check if any valid element had a positive result
   *    if (alpaka::warp::any(acc, flag)) {
   *      // ...
   *    }
   *  }
   *
   * Note that the use of warp-level primitives is usually suitable only for the fastest-looping dimension, `N-1`.
   */

  template <typename TAcc,
            std::size_t Dim,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value >= Dim>>
  class uniform_elements_along {
  public:
    ALPAKA_FN_ACC inline uniform_elements_along(TAcc const& acc)
        : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[Dim]},
          first_{alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[Dim] * elements_},
          stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[Dim] * elements_},
          extent_{stride_} {}

    ALPAKA_FN_ACC inline uniform_elements_along(TAcc const& acc, Idx extent)
        : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[Dim]},
          first_{alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[Dim] * elements_},
          stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[Dim] * elements_},
          extent_{extent} {}

    ALPAKA_FN_ACC inline uniform_elements_along(TAcc const& acc, Idx first, Idx extent)
        : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[Dim]},
          first_{alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[Dim] * elements_ + first},
          stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[Dim] * elements_},
          extent_{extent} {}

    class const_iterator;
    using iterator = const_iterator;

    ALPAKA_FN_ACC inline const_iterator begin() const { return const_iterator(elements_, stride_, extent_, first_); }

    ALPAKA_FN_ACC inline const_iterator end() const { return const_iterator(elements_, stride_, extent_, extent_); }

    class const_iterator {
      friend class uniform_elements_along;

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

  /* uniform_elements
   *
   * `uniform_elements(acc [, first], extent)` returns a one-dimensional iteratable range that spans the element indices
   * from `first` (inclusive) to `extent` (exlusive).
   * If `first` is not specified, it defaults to 0.
   * If `extent` is not specified, it defaults to the kernel grid size.
   *
   * `uniform_elements(acc, ...)` is a shorthand for `uniform_elements_along<0>(acc, ...)`.
   *
   * To cover the problem space, different threads may execute a different number of iterations. As a result, it is not
   * safe to call `alpaka::syncBlockThreads()` and other block-level synchronisations within this loop.
   * If a block synchronisation is needed, one should split the loop into an outer loop over the groups and an inner
   * loop over each group's elements, and synchronise only in the outer loop:
   *
   *  for (auto group : uniform_groups(acc, extent)) {
   *    for (auto element : uniform_group_elements(acc, group, extent)) {
   *       // first part of the computation
   *       // no synchronisations here
   *       ...
   *    }
   *    // wait for all threads to complete the first part
   *    alpaka::syncBlockThreads();
   *    for (auto element : uniform_group_elements(acc, group, extent)) {
   *       // second part of the computation
   *       // no synchronisations here
   *       ...
   *    }
   *    // wait for all threads to complete the second part
   *    alpaka::syncBlockThreads();
   *    ...
   *  }
   *
   * Warp-level primitives require that all threads in the warp execute the same function. If `extent` is not a multiple
   * of the warp size, some of the warps may be incomplete, leading to undefined behaviour - for example, the kernel may
   * hang. To avoid this problem, round up `extent` to a multiple of the warp size, and check the element index
   * explicitly inside the loop:
   *
   *  for (auto element : uniform_elements(acc, round_up_by(extent, alpaka::warp::getSize(acc)))) {
   *    bool flag = false;
   *    if (element < extent) {
   *      // do some work and compute a result flag only for elements up to extent
   *      flag = do_some_work();
   *    }
   *    // check if any valid element had a positive result
   *    if (alpaka::warp::any(acc, flag)) {
   *      // ...
   *    }
   *  }
   *
   * Note that `uniform_elements(acc, ...)` is only suitable for one-dimensional kernels. For N-dimensional kernels, use
   *   - `uniform_elements_nd(acc, ...)` to cover an N-dimensional problem space with a single loop;
   *   - `uniform_elements_along<Dim>(acc, ...)` to perform the iteration explicitly along dimension `Dim`;
   *   - `uniform_elements_x(acc, ...)`, `uniform_elements_y(acc, ...)`, or `uniform_elements_z(acc, ...)` to loop
   *     along the fastest, second-fastest, or third-fastest dimension.
   */

  template <typename TAcc,
            typename... TArgs,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value == 1>>
  ALPAKA_FN_ACC inline auto uniform_elements(TAcc const& acc, TArgs... args) {
    return uniform_elements_along<TAcc, 0>(acc, static_cast<Idx>(args)...);
  }

  /* uniform_elements_x, _y, _z
   *
   * Like `uniform_elements` for N-dimensional kernels, along the fastest, second-fastest, and third-fastest dimensions.
   */

  template <typename TAcc,
            typename... TArgs,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 0)>>
  ALPAKA_FN_ACC inline auto uniform_elements_x(TAcc const& acc, TArgs... args) {
    return uniform_elements_along<TAcc, alpaka::Dim<TAcc>::value - 1>(acc, static_cast<Idx>(args)...);
  }

  template <typename TAcc,
            typename... TArgs,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 1)>>
  ALPAKA_FN_ACC inline auto uniform_elements_y(TAcc const& acc, TArgs... args) {
    return uniform_elements_along<TAcc, alpaka::Dim<TAcc>::value - 2>(acc, static_cast<Idx>(args)...);
  }

  template <typename TAcc,
            typename... TArgs,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 2)>>
  ALPAKA_FN_ACC inline auto uniform_elements_z(TAcc const& acc, TArgs... args) {
    return uniform_elements_along<TAcc, alpaka::Dim<TAcc>::value - 3>(acc, static_cast<Idx>(args)...);
  }

  /* elements_with_stride
   *
   * `elements_with_stride(acc [, first], extent)` returns a one-dimensional iteratable range that spans the element
   * indices from `first` (inclusive) to `extent` (exlusive).
   * If `first` is not specified, it defaults to 0.
   * If `extent` is not specified, it defaults to the kernel grid size.
   *
   * `elements_with_stride(acc, ...)` is a legacy name for `uniform_elements(acc, ...)`.
   */

  template <typename TAcc,
            typename... TArgs,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value == 1>>
  ALPAKA_FN_ACC inline auto elements_with_stride(TAcc const& acc, TArgs... args) {
    return uniform_elements_along<TAcc, 0>(acc, static_cast<Idx>(args)...);
  }

  /* uniform_elements_nd
   *
   * `uniform_elements_nd(acc, extent)` returns an N-dimensional iteratable range that spans the element indices
   * required to cover the given problem size, indicated by `extent`.
   *
   * To cover the problem space, different threads may execute a different number of iterations. As a result, it is not
   * safe to call `alpaka::syncBlockThreads()` and other block-level synchronisations within this loop.
   * If a block synchronisation is needed, one should split the loop into an outer loop over the groups and an inner
   * loop over each group's elements, and synchronise only in the outer loop:
   *
   *  for (auto group0 : uniform_groups_along<0>(acc, extent[0])) {
   *    for (auto group1 : uniform_groups_along<1>(acc, extent[1])) {
   *      for (auto element0 : uniform_group_elements_along<0>(acc, group0, extent[0])) {
   *        for (auto element1 : uniform_group_elements_along<1>(acc, group1, extent[1])) {
   *           // first part of the computation
   *           // no synchronisations here
   *           ...
   *        }
   *      }
   *      // wait for all threads to complete the first part
   *      alpaka::syncBlockThreads();
   *      for (auto element0 : uniform_group_elements_along<0>(acc, group0, extent[0])) {
   *        for (auto element1 : uniform_group_elements_along<1>(acc, group1, extent[1])) {
   *           // second part of the computation
   *           // no synchronisations here
   *           ...
   *        }
   *      }
   *      // wait for all threads to complete the second part
   *      alpaka::syncBlockThreads();
   *      ...
   *    }
   *  }
   *
   * For more details, see `uniform_elements_along<Dim>(acc, ...)`.
   */

  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 0)>>
  class uniform_elements_nd {
  public:
    using Dim = alpaka::Dim<TAcc>;
    using Vec = alpaka::Vec<Dim, Idx>;

    ALPAKA_FN_ACC inline uniform_elements_nd(TAcc const& acc)
        : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)},
          thread_{alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc) * elements_},
          stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc) * elements_},
          extent_{stride_} {}

    ALPAKA_FN_ACC inline uniform_elements_nd(TAcc const& acc, Vec extent)
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
      friend class uniform_elements_nd;

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
      ALPAKA_FN_ACC inline const_iterator(uniform_elements_nd const* loop, Vec first)
          : loop_{loop},
            first_{alpaka::elementwise_min(first, loop->extent_)},
            range_{alpaka::elementwise_min(first + loop->elements_, loop->extent_)},
            index_{first_} {}

      // construct an end iterator, pointing post the end of the extent
      ALPAKA_FN_ACC inline const_iterator(uniform_elements_nd const* loop, at_end_t const&)
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

      // const pointer to the uniform_elements_nd that the iterator refers to
      const uniform_elements_nd* loop_;

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

  /* elements_with_stride_nd
   *
   * `elements_with_stride_nd(acc, extent)` returns an N-dimensional iteratable range that spans the element indices
   * required to cover the given problem size, indicated by `extent`.
   *
   * `elements_with_stride_nd(acc, ...)` is a legacy name for `uniform_elements_nd(acc, ...)`.
   */

  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 0)>>
  ALPAKA_FN_ACC inline auto elements_with_stride_nd(TAcc const& acc) {
    return uniform_elements_nd<TAcc>(acc);
  }

  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 0)>>
  ALPAKA_FN_ACC inline auto elements_with_stride_nd(TAcc const& acc, alpaka::Vec<alpaka::Dim<TAcc>, Idx> extent) {
    return uniform_elements_nd<TAcc>(acc, extent);
  }

  /* uniform_groups_along
   *
   * `uniform_groups_along<Dim>(acc, elements)` returns a one-dimensional iteratable range than spans the group indices
   * required to cover the given problem size along the `Dim` dimension, in units of the block size. `elements`
   * indicates the total number of elements, across all groups; if not specified, it defaults to the kernel grid size
   * along the `Dim` dimension.
   *
   * In a 1-dimensional kernel, `uniform_groups(acc, ...)` is a shorthand for `uniform_groups_along<0>(acc, ...)`.
   *
   * In an N-dimensional kernel, dimension 0 is the one that increases more slowly (e.g. the outer loop), followed by
   * dimension 1, up to dimension N-1 that increases fastest (e.g. the inner loop).
   * For convenience when converting CUDA or HIP code, `uniform_groups_x(acc, ...)`, `_y` and `_z` are shorthands for 
   * `uniform_groups_along<N-1>(acc, ...)`, `<N-2>` and `<N-3>`.
   *
   * `uniform_groups_along<Dim>` should be called consistently by all the threads in a block. All threads in a block see
   * the same loop iterations, while threads in different blocks may see a different number of iterations.
   * If the work division has more blocks than the required number of groups, the first blocks will perform one
   * iteration of the loop, while the other blocks will exit the loop immediately.
   * If the work division has less blocks than the required number of groups, some of the blocks will perform more than
   * one iteration, in order to cover then whole problem space.
   *
   * If the problem size is not a multiple of the block size, the last group will process a number of elements smaller
   * than the block size. However, also in this case all threads in the block will execute the same number of iterations
   * of this loop: this makes it safe to use block-level synchronisations in the loop body. It is left to the inner loop
   * (or the user) to ensure that only the correct number of threads process any data; this logic is implemented by 
   * `uniform_group_elements_along<Dim>(acc, group, elements)`.
   *
   * For example, if the block size is 64 and there are 400 elements
   *
   *   for (auto group: uniform_groups_along<Dim>(acc, 400)
   *
   * will return the group range from 0 to 6, distributed across all blocks in the work division: group 0 should cover
   * the elements from 0 to 63, group 1 should cover the elements from 64 to 127, etc., until the last group, group 6,
   * should cover the elements from 384 to 399. All the threads of the block will process this last group; it is up to
   * the inner loop to not process the non-existing elements after 399.
   *
   * If the work division has more than 7 blocks, the first 7 will perform one iteration of the loop, while the other
   * blocks will exit the loop immediately. For example if the work division has 8 blocks, the blocks from 0 to 6 will
   * process one group while block 7 will no process any.
   *
   * If the work division has less than 7 blocks, some of the blocks will perform more than one iteration of the loop,
   * in order to cover then whole problem space. For example if the work division has 4 blocks, block 0 will process the
   * groups 0 and 4, block 1 will process groups 1 and 5, group 2 will process groups 2 and 6, and block 3 will process
   * group 3.
   *
   * See `uniform_elements_along<Dim>(acc, ...)` for a concrete example using `uniform_groups_along<Dim>` and
   * `uniform_group_elements_along<Dim>`.
   */

  template <typename TAcc,
            std::size_t Dim,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value >= Dim>>
  class uniform_groups_along {
  public:
    ALPAKA_FN_ACC inline uniform_groups_along(TAcc const& acc)
        : first_{alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[Dim]},
          stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[Dim]},
          extent_{stride_} {}

    // extent is the total number of elements (not blocks)
    ALPAKA_FN_ACC inline uniform_groups_along(TAcc const& acc, Idx extent)
        : first_{alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[Dim]},
          stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[Dim]},
          extent_{divide_up_by(extent, alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[Dim])} {}

    class const_iterator;
    using iterator = const_iterator;

    ALPAKA_FN_ACC inline const_iterator begin() const { return const_iterator(stride_, extent_, first_); }

    ALPAKA_FN_ACC inline const_iterator end() const { return const_iterator(stride_, extent_, extent_); }

    class const_iterator {
      friend class uniform_groups_along;

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

  /* uniform_groups
   *
   * `uniform_groups(acc, elements)` returns a one-dimensional iteratable range than spans the group indices required to
   * cover the given problem size, in units of the block size. `elements` indicates the total number of elements, across
   * all groups; if not specified, it defaults to the kernel grid size.
   *
   * `uniform_groups(acc, ...)` is a shorthand for `uniform_groups_along<0>(acc, ...)`.
   *
   * `uniform_groups(acc, ...)` should be called consistently by all the threads in a block. All threads in a block see
   * the same loop iterations, while threads in different blocks may see a different number of iterations.
   * If the work division has more blocks than the required number of groups, the first blocks will perform one
   * iteration of the loop, while the other blocks will exit the loop immediately.
   * If the work division has less blocks than the required number of groups, some of the blocks will perform more than
   * one iteration, in order to cover then whole problem space.
   *
   * If the problem size is not a multiple of the block size, the last group will process a number of elements smaller
   * than the block size. However, also in this case all threads in the block will execute the same number of iterations
   * of this loop: this makes it safe to use block-level synchronisations in the loop body. It is left to the inner loop
   * (or the user) to ensure that only the correct number of threads process any data; this logic is implemented by 
   * `uniform_group_elements(acc, group, elements)`.
   *
   * For example, if the block size is 64 and there are 400 elements
   *
   *   for (auto group: uniform_groups(acc, 400)
   *
   * will return the group range from 0 to 6, distributed across all blocks in the work division: group 0 should cover
   * the elements from 0 to 63, group 1 should cover the elements from 64 to 127, etc., until the last group, group 6,
   * should cover the elements from 384 to 399. All the threads of the block will process this last group; it is up to
   * the inner loop to not process the non-existing elements after 399.
   *
   * If the work division has more than 7 blocks, the first 7 will perform one iteration of the loop, while the other
   * blocks will exit the loop immediately. For example if the work division has 8 blocks, the blocks from 0 to 6 will
   * process one group while block 7 will no process any.
   *
   * If the work division has less than 7 blocks, some of the blocks will perform more than one iteration of the loop,
   * in order to cover then whole problem space. For example if the work division has 4 blocks, block 0 will process the
   * groups 0 and 4, block 1 will process groups 1 and 5, group 2 will process groups 2 and 6, and block 3 will process
   * group 3.
   *
   * See `uniform_elements(acc, ...)` for a concrete example using `uniform_groups` and `uniform_group_elements`.
   *
   * Note that `uniform_groups(acc, ...)` is only suitable for one-dimensional kernels. For N-dimensional kernels, use
   *   - `uniform_groups_along<Dim>(acc, ...)` to perform the iteration explicitly along dimension `Dim`;
   *   - `uniform_groups_x(acc, ...)`, `uniform_groups_y(acc, ...)`, or `uniform_groups_z(acc, ...)` to loop
   *     along the fastest, second-fastest, or third-fastest dimension.
   */

  template <typename TAcc,
            typename... TArgs,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value == 1>>
  ALPAKA_FN_ACC inline auto uniform_groups(TAcc const& acc, TArgs... args) {
    return uniform_groups_along<TAcc, 0>(acc, static_cast<Idx>(args)...);
  }

  /* uniform_groups_x, _y, _z
   *
   * Like `uniform_groups` for N-dimensional kernels, along the fastest, second-fastest, and third-fastest dimensions.
   */

  template <typename TAcc,
            typename... TArgs,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 0)>>
  ALPAKA_FN_ACC inline auto uniform_groups_x(TAcc const& acc, TArgs... args) {
    return uniform_groups_along<TAcc, alpaka::Dim<TAcc>::value - 1>(acc, static_cast<Idx>(args)...);
  }

  template <typename TAcc,
            typename... TArgs,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 1)>>
  ALPAKA_FN_ACC inline auto uniform_groups_y(TAcc const& acc, TArgs... args) {
    return uniform_groups_along<TAcc, alpaka::Dim<TAcc>::value - 2>(acc, static_cast<Idx>(args)...);
  }

  template <typename TAcc,
            typename... TArgs,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 2)>>
  ALPAKA_FN_ACC inline auto uniform_groups_z(TAcc const& acc, TArgs... args) {
    return uniform_groups_along<TAcc, alpaka::Dim<TAcc>::value - 3>(acc, static_cast<Idx>(args)...);
  }

  /* blocks_with_stride
   *
   * `blocks_with_stride(acc, elements)` returns a one-dimensional iteratable range than spans the group indices
   * required to cover the given problem size, in units of the block size. `elements` indicates the total number of
   * elements, across all groups; if not specified, it defaults to the kernel grid size.
   *
   * `blocks_with_stride(acc, ...)` is a legacy name for `uniform_groups(acc, ...)`.
   */

  template <typename TAcc,
            typename... TArgs,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value == 1>>
  ALPAKA_FN_ACC inline auto blocks_with_stride(TAcc const& acc, TArgs... args) {
    return uniform_groups_along<TAcc, 0>(acc, static_cast<Idx>(args)...);
  }

  /* uniform_group_elements_along
   *
   * `uniform_group_elements_along<Dim>(acc, group, elements)` returns a one-dimensional iteratable range that spans all
   * the elements within the given `group` along dimension `Dim`, as obtained from `uniform_groups_along<Dim>`, up to
   * `elements` (exclusive). `elements` indicates the total number of elements across all groups; if not specified, it
   * defaults to the kernel grid size.
   *
   * In a 1-dimensional kernel, `uniform_group_elements(acc, ...)` is a shorthand for
   * `uniform_group_elements_along<0>(acc, ...)`.
   *
   * In an N-dimensional kernel, dimension 0 is the one that increases more slowly (e.g. the outer loop), followed by 
   * dimension 1, up to dimension N-1 that increases fastest (e.g. the inner loop).
   * For convenience when converting CUDA or HIP code, `uniform_group_elements_x(acc, ...)`, `_y` and `_z` are
   * shorthands for `uniform_group_elements_along<N-1>(acc, ...)`, `<N-2>` and `<N-3>`.
   *
   * Iterating over the range yields values of type `ElementIndex`, that provide the `.global` and `.local` indices of
   * the corresponding element. The global index spans a subset of the range from 0 to `elements` (excluded), while the
   * local index spans the range from 0 to the block size (excluded).
   *
   * The loop will perform a number of iterations up to the number of elements per thread, stopping earlier if the
   * global element index reaches `elements`.
   *
   * If the problem size is not a multiple of the block size, different threads may execute a different number of
   * iterations. As a result, it is not safe to call `alpaka::syncBlockThreads()` within this loop. If a block
   * synchronisation is needed, one should split the loop, and synchronise the threads between the loops.
   * See `uniform_elements_along<Dim>(acc, ...)` for a concrete example using `uniform_groups_along<Dim>` and
   * `uniform_group_elements_along<Dim>`.
   *
   * Warp-level primitives require that all threads in the warp execute the same function. If `elements` is not a
   * multiple of the warp size, some of the warps may be incomplete, leading to undefined behaviour - for example, the
   * kernel may hang. To avoid this problem, round up `elements` to a multiple of the warp size, and check the element
   * index explicitly inside the loop:
   *
   *  for (auto element : uniform_group_elements_along<N-1>(acc, group, round_up_by(elements, alpaka::warp::getSize(acc)))) {
   *    bool flag = false;
   *    if (element < elements) {
   *      // do some work and compute a result flag only for the valid elements
   *      flag = do_some_work();
   *    }
   *    // check if any valid element had a positive result
   *    if (alpaka::warp::any(acc, flag)) {
   *      // ...
   *    }
   *  }
   *
   * Note that the use of warp-level primitives is usually suitable only for the fastest-looping dimension, `N-1`.
   */

  template <typename TAcc,
            std::size_t Dim,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value >= Dim>>
  class uniform_group_elements_along {
  public:
    ALPAKA_FN_ACC inline uniform_group_elements_along(TAcc const& acc, Idx block)
        : first_{block * alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[Dim]},
          local_{alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[Dim] *
                 alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[Dim]},
          range_{local_ + alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[Dim]} {}

    ALPAKA_FN_ACC inline uniform_group_elements_along(TAcc const& acc, Idx block, Idx extent)
        : first_{block * alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[Dim]},
          local_{std::min(extent - first_,
                          alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[Dim] *
                              alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[Dim])},
          range_{std::min(extent - first_, local_ + alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[Dim])} {}

    class const_iterator;
    using iterator = const_iterator;

    ALPAKA_FN_ACC inline const_iterator begin() const { return const_iterator(local_, first_, range_); }

    ALPAKA_FN_ACC inline const_iterator end() const { return const_iterator(range_, first_, range_); }

    class const_iterator {
      friend class uniform_group_elements_along;

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

  /* uniform_group_elements
   *
   * `uniform_group_elements(acc, group, elements)` returns a one-dimensional iteratable range that spans all the
   * elements within the given `group`, as obtained from `uniform_groups`, up to `elements` (exclusive). `elements`
   * indicates the total number of elements across all groups; if not specified, it defaults to the kernel grid size.
   *
   * `uniform_group_elements(acc, ...)` is a shorthand for `uniform_group_elements_along<0>(acc, ...)`.
   *
   * Iterating over the range yields values of type `ElementIndex`, that provide the `.global` and `.local` indices of
   * the corresponding element. The global index spans a subset of the range from 0 to `elements` (excluded), while the
   * local index spans the range from 0 to the block size (excluded).
   *
   * The loop will perform a number of iterations up to the number of elements per thread, stopping earlier if the
   * global element index reaches `elements`.
   *
   * If the problem size is not a multiple of the block size, different threads may execute a different number of
   * iterations. As a result, it is not safe to call `alpaka::syncBlockThreads()` within this loop. If a block
   * synchronisation is needed, one should split the loop, and synchronise the threads between the loops.
   * See `uniform_elements(acc, ...)` for a concrete example using `uniform_groups` and `uniform_group_elements`.
   *
   * Warp-level primitives require that all threads in the warp execute the same function. If `elements` is not a
   * multiple of the warp size, some of the warps may be incomplete, leading to undefined behaviour - for example, the
   * kernel may hang. To avoid this problem, round up `elements` to a multiple of the warp size, and check the element
   * index explicitly inside the loop:
   *
   *  for (auto element : uniform_group_elements(acc, group, round_up_by(elements, alpaka::warp::getSize(acc)))) {
   *    bool flag = false;
   *    if (element < elements) {
   *      // do some work and compute a result flag only for the valid elements
   *      flag = do_some_work();
   *    }
   *    // check if any valid element had a positive result
   *    if (alpaka::warp::any(acc, flag)) {
   *      // ...
   *    }
   *  }
   *
   * Note that `uniform_group_elements(acc, ...)` is only suitable for one-dimensional kernels. For N-dimensional
   * kernels, use
   *   - `uniform_group_elements_along<Dim>(acc, ...)` to perform the iteration explicitly along dimension `Dim`;
   *   - `uniform_group_elements_x(acc, ...)`, `uniform_group_elements_y(acc, ...)`, or
   *     `uniform_group_elements_z(acc, ...)` to loop along the fastest, second-fastest, or third-fastest dimension.
   */

  template <typename TAcc,
            typename... TArgs,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value == 1>>
  ALPAKA_FN_ACC inline auto uniform_group_elements(TAcc const& acc, TArgs... args) {
    return uniform_group_elements_along<TAcc, 0>(acc, static_cast<Idx>(args)...);
  }

  /* uniform_group_elements_x, _y, _z
   *
   * Like `uniform_group_elements` for N-dimensional kernels, along the fastest, second-fastest, and third-fastest
   * dimensions.
   */

  template <typename TAcc,
            typename... TArgs,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 0)>>
  ALPAKA_FN_ACC inline auto uniform_group_elements_x(TAcc const& acc, TArgs... args) {
    return uniform_group_elements_along<TAcc, alpaka::Dim<TAcc>::value - 1>(acc, static_cast<Idx>(args)...);
  }

  template <typename TAcc,
            typename... TArgs,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 1)>>
  ALPAKA_FN_ACC inline auto uniform_group_elements_y(TAcc const& acc, TArgs... args) {
    return uniform_group_elements_along<TAcc, alpaka::Dim<TAcc>::value - 2>(acc, static_cast<Idx>(args)...);
  }

  template <typename TAcc,
            typename... TArgs,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 2)>>
  ALPAKA_FN_ACC inline auto uniform_group_elements_z(TAcc const& acc, TArgs... args) {
    return uniform_group_elements_along<TAcc, alpaka::Dim<TAcc>::value - 3>(acc, static_cast<Idx>(args)...);
  }

  /* elements_in_block
   *
   * `elements_in_block(acc, group, elements)` returns a one-dimensional iteratable range that spans all the elements
   * within the given `group`, as obtained from `uniform_groups`, up to `elements` (exclusive). `elements` indicates the
   * total number of elements across all groups; if not specified, it defaults to the kernel grid size.
   *
   * `elements_in_block(acc, ...)` is a legacy for `uniform_group_elements(acc, ...)`.
   */

  template <typename TAcc,
            typename... TArgs,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value == 1>>
  ALPAKA_FN_ACC inline auto elements_in_block(TAcc const& acc, TArgs... args) {
    return uniform_group_elements_along<TAcc, 0>(acc, static_cast<Idx>(args)...);
  }

  /* independent_groups_along
   *
   * `independent_groups_along<Dim>(acc, groups)` returns a one-dimensional iteratable range than spans the group
   * indices from 0 to `groups`; the groups are assigned to the blocks along the `Dim` dimension. If `groups` is not
   * specified, it defaults to the number of blocks along the `Dim` dimension.
   *
   * In a 1-dimensional kernel, `independent_groups(acc, ...)` is a shorthand for
   * `independent_groups_along<0>(acc, ...)`.
   *
   * In an N-dimensional kernel, dimension 0 is the one that increases more slowly (e.g. the outer loop), followed by
   * dimension 1, up to dimension N-1 that increases fastest (e.g. the inner loop).
   * For convenience when converting CUDA or HIP code, `independent_groups_x(acc, ...)`, `_y` and `_z` are shorthands
   * for `independent_groups_along<N-1>(acc, ...)`, `<N-2>` and `<N-3>`.
   *
   * `independent_groups_along<Dim>` should be called consistently by all the threads in a block. All threads in a block
   * see the same loop iterations, while threads in different blocks may see a different number of iterations.
   * If the work division has more blocks than the required number of groups, the first blocks will perform one
   * iteration of the loop, while the other blocks will exit the loop immediately.
   * If the work division has less blocks than the required number of groups, some of the blocks will perform more than
   * one iteration, in order to cover then whole problem space.
   *
   * For example,
   *
   *   for (auto group: independent_groups_along<Dim>(acc, 7))
   *
   * will return the group range from 0 to 6, distributed across all blocks in the work division.
   * If the work division has more than 7 blocks, the first 7 will perform one iteration of the loop, while the other
   * blocks will exit the loop immediately. For example if the work division has 8 blocks, the blocks from 0 to 6 will
   * process one group while block 7 will no process any.
   * If the work division has less than 7 blocks, some of the blocks will perform more than one iteration of the loop,
   * in order to cover then whole problem space. For example if the work division has 4 blocks, block 0 will process the
   * groups 0 and 4, block 1 will process groups 1 and 5, group 2 will process groups 2 and 6, and block 3 will process
   * group 3.
   */

  template <typename TAcc,
            std::size_t Dim,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value >= Dim>>
  class independent_groups_along {
  public:
    ALPAKA_FN_ACC inline independent_groups_along(TAcc const& acc)
        : first_{alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[Dim]},
          stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[Dim]},
          extent_{stride_} {}

    ALPAKA_FN_ACC inline independent_groups_along(TAcc const& acc, Idx groups)
        : first_{alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[Dim]},
          stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[Dim]},
          extent_{groups} {}

    class const_iterator;
    using iterator = const_iterator;

    ALPAKA_FN_ACC inline const_iterator begin() const { return const_iterator(stride_, extent_, first_); }

    ALPAKA_FN_ACC inline const_iterator end() const { return const_iterator(stride_, extent_, extent_); }

    class const_iterator {
      friend class independent_groups_along;

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

  /* independent_groups
   *
   * `independent_groups(acc, groups)` returns a one-dimensional iteratable range than spans the group indices from 0 to
   * `groups`. If `groups` is not specified, it defaults to the number of blocks.
   *
   * `independent_groups(acc, ...)` is a shorthand for `independent_groups_along<0>(acc, ...)`.
   *
   * `independent_groups(acc, ...)` should be called consistently by all the threads in a block. All threads in a block
   * see the same loop iterations, while threads in different blocks may see a different number of iterations.
   * If the work division has more blocks than the required number of groups, the first blocks will perform one
   * iteration of the loop, while the other blocks will exit the loop immediately.
   * If the work division has less blocks than the required number of groups, some of the blocks will perform more than
   * one iteration, in order to cover then whole problem space.
   *
   * For example,
   *
   *   for (auto group: independent_groups(acc, 7))
   *
   * will return the group range from 0 to 6, distributed across all blocks in the work division.
   * If the work division has more than 7 blocks, the first 7 will perform one iteration of the loop, while the other
   * blocks will exit the loop immediately. For example if the work division has 8 blocks, the blocks from 0 to 6 will
   * process one group while block 7 will no process any.
   * If the work division has less than 7 blocks, some of the blocks will perform more than one iteration of the loop,
   * in order to cover then whole problem space. For example if the work division has 4 blocks, block 0 will process the
   * groups 0 and 4, block 1 will process groups 1 and 5, group 2 will process groups 2 and 6, and block 3 will process
   * group 3.
   *
   * Note that `independent_groups(acc, ...)` is only suitable for one-dimensional kernels. For N-dimensional kernels,
   * use
   *   - `independent_groups_along<Dim>(acc, ...)` to perform the iteration explicitly along dimension `Dim`;
   *   - `independent_groups_x(acc, ...)`, `independent_groups_y(acc, ...)`, or `independent_groups_z(acc, ...)` to loop
   *     along the fastest, second-fastest, or third-fastest dimension.
   */

  template <typename TAcc,
            typename... TArgs,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value == 1>>
  ALPAKA_FN_ACC inline auto independent_groups(TAcc const& acc, TArgs... args) {
    return independent_groups_along<TAcc, 0>(acc, static_cast<Idx>(args)...);
  }

  /* independent_groups_x, _y, _z
   *
   * Like `independent_groups` for N-dimensional kernels, along the fastest, second-fastest, and third-fastest
   * dimensions.
   */

  template <typename TAcc,
            typename... TArgs,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 0)>>
  ALPAKA_FN_ACC inline auto independent_groups_x(TAcc const& acc, TArgs... args) {
    return independent_groups_along<TAcc, alpaka::Dim<TAcc>::value - 1>(acc, static_cast<Idx>(args)...);
  }

  template <typename TAcc,
            typename... TArgs,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 1)>>
  ALPAKA_FN_ACC inline auto independent_groups_y(TAcc const& acc, TArgs... args) {
    return independent_groups_along<TAcc, alpaka::Dim<TAcc>::value - 2>(acc, static_cast<Idx>(args)...);
  }

  template <typename TAcc,
            typename... TArgs,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 2)>>
  ALPAKA_FN_ACC inline auto independent_groups_z(TAcc const& acc, TArgs... args) {
    return independent_groups_along<TAcc, alpaka::Dim<TAcc>::value - 3>(acc, static_cast<Idx>(args)...);
  }

  /* independent_group_elements_along
   */

  template <typename TAcc,
            std::size_t Dim,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value >= Dim>>
  class independent_group_elements_along {
  public:
    ALPAKA_FN_ACC inline independent_group_elements_along(TAcc const& acc)
        : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[Dim]},
          thread_{alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[Dim] * elements_},
          stride_{alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[Dim] * elements_},
          extent_{stride_} {}

    ALPAKA_FN_ACC inline independent_group_elements_along(TAcc const& acc, Idx extent)
        : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[Dim]},
          thread_{alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[Dim] * elements_},
          stride_{alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[Dim] * elements_},
          extent_{extent} {}

    ALPAKA_FN_ACC inline independent_group_elements_along(TAcc const& acc, Idx first, Idx extent)
        : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[Dim]},
          thread_{alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[Dim] * elements_ + first},
          stride_{alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[Dim] * elements_},
          extent_{extent} {}

    class const_iterator;
    using iterator = const_iterator;

    ALPAKA_FN_ACC inline const_iterator begin() const { return const_iterator(elements_, stride_, extent_, thread_); }

    ALPAKA_FN_ACC inline const_iterator end() const { return const_iterator(elements_, stride_, extent_, extent_); }

    class const_iterator {
      friend class independent_group_elements_along;

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

  /* independent_group_elements
   */

  template <typename TAcc,
            typename... TArgs,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value == 1>>
  ALPAKA_FN_ACC inline auto independent_group_elements(TAcc const& acc, TArgs... args) {
    return independent_group_elements_along<TAcc, 0>(acc, static_cast<Idx>(args)...);
  }

  /* independent_group_elements_x, _y, _z
   *
   * Like `independent_group_elements` for N-dimensional kernels, along the fastest, second-fastest, and third-fastest
   * dimensions.
   */

  template <typename TAcc,
            typename... TArgs,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 0)>>
  ALPAKA_FN_ACC inline auto independent_group_elements_x(TAcc const& acc, TArgs... args) {
    return independent_group_elements_along<TAcc, alpaka::Dim<TAcc>::value - 1>(acc, static_cast<Idx>(args)...);
  }

  template <typename TAcc,
            typename... TArgs,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 1)>>
  ALPAKA_FN_ACC inline auto independent_group_elements_y(TAcc const& acc, TArgs... args) {
    return independent_group_elements_along<TAcc, alpaka::Dim<TAcc>::value - 2>(acc, static_cast<Idx>(args)...);
  }

  template <typename TAcc,
            typename... TArgs,
            typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 2)>>
  ALPAKA_FN_ACC inline auto independent_group_elements_z(TAcc const& acc, TArgs... args) {
    return independent_group_elements_along<TAcc, alpaka::Dim<TAcc>::value - 3>(acc, static_cast<Idx>(args)...);
  }

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
