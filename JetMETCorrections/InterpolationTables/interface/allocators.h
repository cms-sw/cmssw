#ifndef NPSTAT_ALLOCATORS_HH_
#define NPSTAT_ALLOCATORS_HH_

/*!
// \file allocators.h
//
// \brief Utilities related to memory management
//
// Author: I. Volobouev
//
// October 2009
*/

#include <cassert>

namespace npstat {
  /**
    // Function for allocating memory buffers if their size
    // exceeds the size of the buffer available on the stack
    */
  template <typename T>
  inline T* makeBuffer(unsigned sizeNeeded, T* stackBuffer, unsigned sizeofStackBuffer) {
    if (sizeNeeded > sizeofStackBuffer || stackBuffer == nullptr)
      return new T[sizeNeeded];
    else
      return stackBuffer;
  }

  /** Function for freeing memory buffers allocated by "makeBuffer" */
  template <typename T>
  inline void destroyBuffer(T* thisBuffer, const T* stackBuffer) {
    if (thisBuffer != stackBuffer)
      delete[] thisBuffer;
  }

  /** Copy a buffer (with possible type conversion on the fly) */
  template <typename T1, typename T2>
  inline void copyBuffer(T1* dest, const T2* source, const unsigned long len) {
    if (len) {
      assert(dest);
      assert(source);
      for (unsigned long i = 0; i < len; ++i)
        *dest++ = static_cast<T1>(*source++);
    }
  }

  /**
    // Copy a buffer (with possible type conversion on the fly)
    // transposing it in the process (treating as a square matrix)
    */
  template <typename T1, typename T2>
  inline void transposeBuffer(T1* dest, const T2* source, const unsigned long dim) {
    if (dim) {
      assert(dest);
      assert(source);
      for (unsigned long i = 0; i < dim; ++i) {
        for (unsigned long j = 0; j < dim; ++j)
          dest[j * dim] = static_cast<T1>(*source++);
        ++dest;
      }
    }
  }

  /**
    // Copy a buffer (with possible type conversion on the fly)
    // transposing it in the process (treating as an M x N matrix)
    */
  template <typename T1, typename T2>
  inline void transposeBuffer(T1* dest, const T2* source, const unsigned long M, const unsigned long N) {
    if (M && N) {
      assert(dest);
      assert(source);
      for (unsigned long i = 0; i < M; ++i) {
        for (unsigned long j = 0; j < N; ++j)
          dest[j * M] = static_cast<T1>(*source++);
        ++dest;
      }
    }
  }

  /** 
    // Clear a buffer (set all elements to the value produced by the
    // default constructor)
    */
  template <typename T>
  inline void clearBuffer(T* buf, const unsigned long len) {
    if (len) {
      assert(buf);
      const T zero = T();
      for (unsigned long i = 0; i < len; ++i)
        *buf++ = zero;
    }
  }
}  // namespace npstat

#endif  // NPSTAT_ALLOCATORS_HH_
