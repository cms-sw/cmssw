//
// Author: Felice Pantaleo, CERN
//

#ifndef GPU_VECARRAY_H_
#define GPU_VECARRAY_H_

#include <cuda.h>
#include <cuda_runtime.h>

namespace GPU {

template <class T, int maxSize> struct VecArray {
  __inline__ constexpr int push_back_unsafe(const T &element) {
    auto previousSize = m_size;
    m_size++;
    if (previousSize < maxSize) {
      m_data[previousSize] = element;
      return previousSize;
    } else {
      --m_size;
      return -1;
    }
  }

  template <class... Ts> constexpr int emplace_back_unsafe(Ts &&... args) {
    auto previousSize = m_size;
    m_size++;
    if (previousSize < maxSize) {
      (new (&m_data[previousSize]) T(std::forward<Ts>(args)...));
      return previousSize;
    } else {
      --m_size;
      return -1;
    }
  }

  __inline__ constexpr T & back() const {
    if (m_size > 0) {
      return m_data[m_size - 1];
    } else
      return T(); //undefined behaviour
  }


  #if defined(__NVCC__) || defined(__CUDACC__)
    // thread-safe version of the vector, when used in a CUDA kernel
    __device__ int push_back(const T &element) {
      auto previousSize = atomicAdd(&m_size, 1);
      if (previousSize < maxSize) {
        m_data[previousSize] = element;
        return previousSize;
      } else {
        atomicSub(&m_size, 1);
        return -1;
      }
    }

    template <class... Ts> __device__ int emplace_back(Ts &&... args) {
      auto previousSize = atomicAdd(&m_size, 1);
      if (previousSize < maxSize) {
        (new (&m_data[previousSize]) T(std::forward<Ts>(args)...));
        return previousSize;
      } else {
        atomicSub(&m_size, 1);
        return -1;
      }
    }
  #endif

  __inline__ __host__ __device__ T pop_back() {
    if (m_size > 0) {
      auto previousSize = m_size--;
      return m_data[previousSize - 1];
    } else
      return T();
  }

  __inline__ constexpr int size() const { return m_size; }

  __inline__ constexpr T& operator[](int i) { return m_data[i]; }

  __inline__ constexpr const T& operator[](int i) const { return m_data[i]; }

  __inline__ constexpr void reset() { m_size = 0; }

  __inline__ constexpr int capacity() const { return maxSize; }

  __inline__ constexpr T *data() const { return m_data; }

  __inline__ constexpr void resize(int size) { m_size = size; }

  __inline__ constexpr bool empty() const { return 0 == m_size; }

  __inline__ constexpr bool full() const { return maxSize == m_size; }

  int m_size = 0;

  T m_data[maxSize];
};

}

#endif // GPU_VECARRAY_H_
