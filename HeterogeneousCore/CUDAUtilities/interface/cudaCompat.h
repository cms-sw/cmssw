#ifndef HeterogeneousCore_CUDAUtilities_interface_cudaCompat_h
#define HeterogeneousCore_CUDAUtilities_interface_cudaCompat_h

/*
 * Everything you need to run cuda code in plain sequential c++ code
 */

#ifndef __CUDACC__

#include <algorithm>
#include <cstdint>
#include <cstring>

// include the CUDA runtime header to define some of the attributes, types and sybols also on the CPU
#include <cuda_runtime.h>

// make sure function are inlined to avoid multiple definition
#undef __global__
#define __global__ inline __attribute__((always_inline))

#undef __forceinline__
#define __forceinline__ inline __attribute__((always_inline))

#undef __launch_bounds__
#define __launch_bounds__(...)

namespace cms {
  namespace cudacompat {

    // run serially with a single thread
    // 1-dimensional block
    const dim3 threadIdx = {0, 0, 0};
    const dim3 blockDim = {1, 1, 1};
    // 1-dimensional grid
    const dim3 blockIdx = {0, 0, 0};
    const dim3 gridDim = {1, 1, 1};

    template <typename T1, typename T2>
    T1 atomicCAS(T1* address, T1 compare, T2 val) {
      T1 old = *address;
      *address = old == compare ? val : old;
      return old;
    }

    template <typename T1, typename T2>
    T1 atomicCAS_block(T1* address, T1 compare, T2 val) {
      return atomicCAS(address, compare, val);
    }

    template <typename T1, typename T2>
    T1 atomicInc(T1* a, T2 b) {
      auto ret = *a;
      if ((*a) < T1(b))
        (*a)++;
      return ret;
    }

    template <typename T1, typename T2>
    T1 atomicInc_block(T1* a, T2 b) {
      return atomicInc(a, b);
    }

    template <typename T1, typename T2>
    T1 atomicAdd(T1* a, T2 b) {
      auto ret = *a;
      (*a) += b;
      return ret;
    }

    template <typename T1, typename T2>
    T1 atomicAdd_block(T1* a, T2 b) {
      return atomicAdd(a, b);
    }

    template <typename T1, typename T2>
    T1 atomicSub(T1* a, T2 b) {
      auto ret = *a;
      (*a) -= b;
      return ret;
    }

    template <typename T1, typename T2>
    T1 atomicSub_block(T1* a, T2 b) {
      return atomicSub(a, b);
    }

    template <typename T1, typename T2>
    T1 atomicMin(T1* a, T2 b) {
      auto ret = *a;
      *a = std::min(*a, T1(b));
      return ret;
    }

    template <typename T1, typename T2>
    T1 atomicMin_block(T1* a, T2 b) {
      return atomicMin(a, b);
    }

    template <typename T1, typename T2>
    T1 atomicMax(T1* a, T2 b) {
      auto ret = *a;
      *a = std::max(*a, T1(b));
      return ret;
    }

    template <typename T1, typename T2>
    T1 atomicMax_block(T1* a, T2 b) {
      return atomicMax(a, b);
    }

    inline void __syncthreads() {}
    inline bool __syncthreads_or(bool x) { return x; }
    inline bool __syncthreads_and(bool x) { return x; }

    inline void __trap() { abort(); }
    inline void __threadfence() {}

    template <typename T>
    inline T __ldg(T const* x) {
      return *x;
    }

    namespace cooperative_groups {

      // This class represents the grid group
     class grid_group {
     private: 
       grid_group() = default;

       friend grid_group this_grid();

      public:
        // Synchronize the threads named in the group.
        // On the serial CPU implementation, do nothing.
        static void sync() {}
     };

     inline grid_group this_grid() { return grid_group{};}

      // This class represents the thread block
      class thread_block {
      private:
        thread_block() = default;

        friend thread_block this_thread_block();

      public:
        // Synchronize the threads named in the group.
        // On the serial CPU implementation, do nothing.
        static void sync() {}

        // Total number of threads in the group.
        // On the serial CPU implementation, always 1.
        static unsigned long long size() { return 1; }

        // Rank of the calling thread within [0, size-1].
        // On the serial CPU implementation, always 0.
        static unsigned long long thread_rank() { return 0; }

        // 3-Dimensional index of the block within the launched grid.
        // On the serial CPU implementation, always {0, 0, 0}.
        static dim3 group_index() { return blockIdx; }

        // 3-Dimensional index of the thread within the launched block
        // On the serial CPU implementation, always {0, 0, 0}.
        static dim3 thread_index() { return threadIdx; }

        // Dimensions of the launched block.
        // On the serial CPU implementation, always {1, 1, 1}.
        static dim3 group_dim() { return blockDim; }
      };

      // Return the current thread block
      inline thread_block this_thread_block() { return thread_block{}; }

      // Represent a tiled group of threads, with compile-time fixed size.
      // On the serial CPU implementation, the only valid Size is 1
      template <unsigned int Size, typename ParentT>
      class thread_block_tile {
      private:
        static_assert(
            Size == 1,
            "The cudaCompat Cooperative Groups implementation supports only tiled groups of a single thread.");

        thread_block_tile() = default;

        friend thread_block_tile<Size, ParentT> tiled_partition(const ParentT& g);

      public:
        // Synchronize the threads named in the group.
        // On the serial CPU implementation, do nothing.
        void sync() const {}

        // Total number of threads in the group.
        // On the serial CPU implementation, always 1.
        unsigned long long size() const { return 1; }

        // Rank of the calling thread within [0, size-1].
        // On the serial CPU implementation, always 0.
        unsigned long long thread_rank() const { return 0; }

        // Returns the number of groups created when the parent group was partitioned.
        // On the serial CPU implementation, always 1.
        unsigned long long meta_group_size() const { return 1; }

        // Linear rank of the group within the set of tiles partitioned from a parent group (bounded by meta_group_size).
        // On the serial CPU implementation, always 0.
        unsigned long long meta_group_rank() const { return 0; }

        // Not implemented - Refer to Warp Shuffle Functions
        template <typename T>
        T shfl(T var, unsigned int src_rank) const;
        template <typename T>
        T shfl_up(T var, int delta) const;
        template <typename T>
        T shfl_down(T var, int delta) const;
        template <typename T>
        T shfl_xor(T var, int delta) const;

        // Not implemented - Refer to Warp Vote Functions
        template <typename T>
        T any(int predicate) const;
        template <typename T>
        T all(int predicate) const;
        template <typename T>
        T ballot(int predicate) const;

        // Not implemented - Refer to Warp Match Functions
        template <typename T>
        T match_any(T val) const;
        template <typename T>
        T match_all(T val, int& pred) const;
      };

      template <unsigned int Size, typename ParentT>
      inline thread_block_tile<Size, ParentT> tiled_partition(const ParentT& g) {
        static_assert(
            Size == 1,
            "The cudaCompat Cooperative Groups implementation supports only tiled groups of a single thread.");

        return thread_block_tile<Size, ParentT>{};
      }

    }  // namespace cooperative_groups

  }  // namespace cudacompat
}  // namespace cms

// make the cudacompat implementation available in the global namespace
using namespace cms::cudacompat;

#endif  // __CUDACC__

#endif  // HeterogeneousCore_CUDAUtilities_interface_cudaCompat_h
