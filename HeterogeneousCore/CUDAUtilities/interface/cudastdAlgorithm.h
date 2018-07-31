#ifndef HeterogeneousCore_CUDAUtilities_cudastdAlgorithm_h
#define HeterogeneousCore_CUDAUtilities_cudastdAlgorithm_h

#include <utility>

#include <cuda_runtime.h>

// reimplementation of std algorithms able to compile with CUDA and run on GPUs,
// mostly by declaringthem constexpr

namespace cuda_std  {

  template<typename T = void>
  struct less {
    __host__ __device__
    constexpr bool operator()(const T &lhs, const T &rhs) const {
      return lhs < rhs;
    }
  };

  template<>
  struct less<void> {
    template<typename T, typename U>
    __host__ __device__
    constexpr bool operator()(const T &lhs, const U &rhs ) const { return lhs < rhs;}
  };

  template<typename RandomIt, typename T, typename Compare = less<T>>
  __host__ __device__
  constexpr
  RandomIt lower_bound(RandomIt first, RandomIt last, const T& value, Compare comp={})
  {
    auto count = last - first;

    while (count > 0) {
        auto it = first;
        auto step = count / 2;
        it += step;
        if (comp(*it, value)) {
            first = ++it;
            count -= step + 1;
        }
        else {
            count = step;
        }
    }
    return first;
  }

  template<typename RandomIt, typename T, typename Compare = less<T>>
  __host__ __device__
  constexpr
  RandomIt upper_bound(RandomIt first, RandomIt last, const T& value, Compare comp={})
  {
    auto count = last - first;

    while (count > 0) {
        auto it = first;
        auto step = count / 2;
        it+=step;
        if (!comp(value,*it)) {
            first = ++it;
            count -= step + 1;
        }
        else {
            count = step;
        }
    }
    return first;
  }

  template<typename RandomIt, typename T, typename Compare = cuda_std::less<T>>
  __host__ __device__
  constexpr
  RandomIt binary_find(RandomIt first, RandomIt last, const T& value, Compare comp={})
  {
    first = cuda_std::lower_bound(first, last, value, comp);
    return first != last && !comp(value, *first) ? first : last;
  }

}

#endif // HeterogeneousCore_CUDAUtilities_cudastdAlgorithm_h
