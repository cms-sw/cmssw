#ifndef FWCore_Utilities_get_underlying_safe_h
#define FWCore_Utilities_get_underlying_safe_h

/*
 Description:

 The function get_underlying(), provided by propagate_const, should not be called directly
 by users because it may cast away constness.
 This header  provides helper function(s) get_underlying_safe() to users of propagate_const<T>.
 The get_underlying_safe() functions avoid this issue.

 If called with a non-const ref argument, get_underlying_safe() simply calls get_underlying().
 If called with a const ref argument, get_underlying_safe() returns a pointer to const,
 which preserves constness, but requires copying the pointer.

 For non-copyable pointers, such as std::unique_ptr, it is not possible to preserve constness.
 so get_underlying_safe() will fail to compile if called with a const ref to a unique_ptr.
 This is intentional.

 This header can be expanded to support other smart pointers as needed.
*/

//
// Original Author:  Bill Tanenbaum
//

// system include files
#include <memory>

// user include files

#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/propagate_const_array.h"

// forward declarations

namespace edm {

  // for std::shared_ptr
  template <typename T>
  constexpr std::shared_ptr<T>& get_underlying_safe(propagate_const<std::shared_ptr<T>>& iP) {
    return get_underlying(iP);
  }
  template <typename T>
  constexpr std::shared_ptr<T const> get_underlying_safe(propagate_const<std::shared_ptr<T>> const& iP) {
    std::shared_ptr<T const> copy = get_underlying(iP);
    return copy;
  }

  template <typename T>
  constexpr std::shared_ptr<T[]>& get_underlying_safe(propagate_const_array<std::shared_ptr<T[]>>& iP) {
    return get_underlying(iP);
  }
  template <typename T>
  constexpr std::shared_ptr<T const[]> get_underlying_safe(propagate_const_array<std::shared_ptr<T[]>> const& iP) {
    std::shared_ptr<T const[]> copy = get_underlying(iP);
    return copy;
  }

  // for bare pointer
  template <typename T>
  constexpr T*& get_underlying_safe(propagate_const<T*>& iP) {
    return get_underlying(iP);
  }
  template <typename T>
  constexpr T const* get_underlying_safe(propagate_const<T*> const& iP) {
    T const* copy = get_underlying(iP);
    return copy;
  }

  template <typename T>
  constexpr T* get_underlying_safe(propagate_const_array<T[]>& iP) {
    return get_underlying(iP);
  }
  template <typename T>
  constexpr T const* get_underlying_safe(propagate_const_array<T[]> const& iP) {
    T const* copy = get_underlying(iP);
    return copy;
  }

  // for std::unique_ptr
  template <typename T>
  constexpr std::unique_ptr<T>& get_underlying_safe(propagate_const<std::unique_ptr<T>>& iP) {
    return get_underlying(iP);
  }
  // the template below will deliberately not compile.
  template <typename T>
  constexpr std::unique_ptr<T const> get_underlying_safe(propagate_const<std::unique_ptr<T>> const& iP) {
    std::unique_ptr<T const> copy = get_underlying(iP);
    return copy;
  }

  template <typename T>
  constexpr std::unique_ptr<T[]>& get_underlying_safe(propagate_const_array<std::unique_ptr<T[]>>& iP) {
    return get_underlying(iP);
  }
  // the template below will deliberately not compile.
  template <typename T>
  constexpr std::unique_ptr<T const[]> get_underlying_safe(propagate_const_array<std::unique_ptr<T[]>> const& iP) {
    std::unique_ptr<T const[]> copy = get_underlying(iP);
    return copy;
  }

}  // namespace edm

#endif
