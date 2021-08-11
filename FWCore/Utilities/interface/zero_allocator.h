#ifndef FWCore_Utilities_zero_allocator_h
#define FWCore_Utilities_zero_allocator_h
/*
    Copyright (c) 2005-2020 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include "tbb/tbb_allocator.h"
#include <cstring>

/* Copied from tbb_2020 branch's tbb/tbb_allocator linked here
   https://github.com/oneapi-src/oneTBB/blob/tbb_2020/include/tbb/tbb_allocator.h
   and renamed to edm namespace because it was removed from oneapi_2021 branch's
   tbb/tbb_allocator.
 */

namespace edm {
  template <typename T, template <typename X> class Allocator = tbb::tbb_allocator>
  class zero_allocator : public Allocator<T> {
  public:
    using value_type = T;
    using base_allocator_type = Allocator<T>;
    template <typename U>
    struct rebind {
      typedef zero_allocator<U, Allocator> other;
    };

    zero_allocator() throw() {}
    zero_allocator(const zero_allocator &a) throw() : base_allocator_type(a) {}
    template <typename U>
    zero_allocator(const zero_allocator<U> &a) throw() : base_allocator_type(Allocator<U>(a)) {}

    T *allocate(const std::size_t n, const void *hint = nullptr) {
      //T* ptr = base_allocator_type::allocate( n, hint );
      T *ptr = base_allocator_type::allocate(n);
      std::memset(static_cast<void *>(ptr), 0, n * sizeof(value_type));
      return ptr;
    }
  };

  template <typename T1, template <typename X1> class B1, typename T2, template <typename X2> class B2>
  inline bool operator==(const zero_allocator<T1, B1> &a, const zero_allocator<T2, B2> &b) {
    return static_cast<B1<T1> >(a) == static_cast<B2<T2> >(b);
  }
  template <typename T1, template <typename X1> class B1, typename T2, template <typename X2> class B2>
  inline bool operator!=(const zero_allocator<T1, B1> &a, const zero_allocator<T2, B2> &b) {
    return static_cast<B1<T1> >(a) != static_cast<B2<T2> >(b);
  }
}  // namespace edm
#endif
