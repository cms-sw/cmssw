#ifndef DataFormats_Common_interface_StdArray_h
#define DataFormats_Common_interface_StdArray_h

#include <array>
#include <cstddef>
#include <iostream>
#include <iterator>

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"

namespace edm {

  // Due to a ROOT limitation an std::array cannot be serialised to a ROOT file.
  // See https://github.com/root-project/root/issues/12007 for a discussion on the issue.
  //
  // This class reimplements the full std::array<T,N> interface, using a regular
  // Reflex dictionary for the ROOT serialisation.
  // To be more GPU-friendly, all methods are constexpr, and out-of-bound data access
  // aborts instead of throwing an exception.
  //
  // Note: dictonaries for edm::StdArray<T,N> where T is a standard C/C++ type
  // should be declared in DataFormats/Common/src/classed_def.xml.

  namespace detail {
    template <typename T, std::size_t N>
    class StdArrayTrait {
    public:
      using array_type = T[N];
    };

    template <typename T>
    class StdArrayTrait<T, 0> {
    public:
      struct array_type {};
    };
  }  // namespace detail

  template <typename T, std::size_t N>
  class StdArray {
  public:
    // Member types

    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using const_reference = value_type const&;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using iterator = pointer;
    using const_iterator = const_pointer;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    // Interoperability with std::array

    // copy assignment from an std::array
    constexpr StdArray& operator=(std::array<T, N> const& init) {
      for (size_type i = 0; i < N; ++i) {
        data_[i] = init[i];
      }
      return *this;
    }

    // move assignment from an std::array
    constexpr StdArray& operator=(std::array<T, N>&& init) {
      for (size_type i = 0; i < N; ++i) {
        data_[i] = std::move(init[i]);
      }
      return *this;
    }

    // cast operator to an std::array
    constexpr operator std::array<T, N>() const {
      std::array<T, N> copy;
      for (size_type i = 0; i < N; ++i) {
        copy[i] = data_[i];
      }
      return copy;
    }

    // Element access

    // Returns a reference to the element at specified location pos, with bounds checking.
    // If pos is not within the range of the container, the program aborts.
    constexpr reference at(size_type pos) {
      if (pos >= N)
        abort();
      return data_[pos];
    }
    constexpr const_reference at(size_type pos) const {
      if (pos >= N)
        abort();
      return data_[pos];
    }

    // Returns a reference to the element at specified location pos. No bounds checking is performed.
    constexpr reference operator[](size_type pos) { return data_[pos]; }
    constexpr const_reference operator[](size_type pos) const { return data_[pos]; }

    // Returns a reference to the first element in the container.
    // Calling front on an empty container causes the program to abort.
    constexpr reference front() {
      if constexpr (N == 0)
        abort();
      return data_[0];
    }
    constexpr const_reference front() const {
      if constexpr (N == 0)
        abort();
      return data_[0];
    }

    // Returns a reference to the last element in the container.
    // Calling back on an empty container causes the program to abort.
    constexpr reference back() {
      if constexpr (N == 0)
        abort();
      return data_[N - 1];
    }
    constexpr const_reference back() const {
      if constexpr (N == 0)
        abort();
      return data_[N - 1];
    }

    // Returns pointer to the underlying array serving as element storage.
    // The pointer is such that range [data(), data() + size()) is always a valid range,
    // even if the container is empty (data() is not dereferenceable in that case).
    constexpr pointer data() noexcept {
      if constexpr (N != 0)
        return data_;
      else
        return nullptr;
    }
    constexpr const_pointer data() const noexcept {
      if constexpr (N != 0)
        return data_;
      else
        return nullptr;
    }

    // Iterators

    // Returns an iterator to the first element of the array.
    // If the array is empty, the returned iterator will be equal to end().
    constexpr iterator begin() noexcept {
      if constexpr (N != 0)
        return data_;
      else
        return nullptr;
    }
    constexpr const_iterator begin() const noexcept {
      if constexpr (N != 0)
        return data_;
      else
        return nullptr;
    }
    constexpr const_iterator cbegin() const noexcept {
      if constexpr (N != 0)
        return data_;
      else
        return nullptr;
    }

    // Returns an iterator to the element following the last element of the array.
    // This element acts as a placeholder; attempting to access it results in undefined behavior.
    constexpr iterator end() noexcept {
      if constexpr (N != 0)
        return data_ + N;
      else
        return nullptr;
    }
    constexpr const_iterator end() const noexcept {
      if constexpr (N != 0)
        return data_ + N;
      else
        return nullptr;
    }
    constexpr const_iterator cend() const noexcept {
      if constexpr (N != 0)
        return data_ + N;
      else
        return nullptr;
    }

    // Returns a reverse iterator to the first element of the reversed array.
    // It corresponds to the last element of the non-reversed array. If the array is empty, the returned iterator is equal to rend().
    constexpr reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
    constexpr const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
    constexpr const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }

    // Returns a reverse iterator to the element following the last element of the reversed array.
    // It corresponds to the element preceding the first element of the non-reversed array. This element acts as a placeholder, attempting to access it results in undefined behavior.
    constexpr reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
    constexpr const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
    constexpr const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }

    // Capacity

    // Checks if the container has no elements, i.e. whether begin() == end().
    constexpr bool empty() const noexcept { return N == 0; }

    // Returns the number of elements in the container, i.e. std::distance(begin(), end()).
    constexpr size_type size() const noexcept { return N; }

    // Returns the maximum number of elements the container is able to hold due to system or library implementation limitations, i.e. std::distance(begin(), end()) for the largest container.
    constexpr size_type max_size() const noexcept { return N; }

    // Operations

    // Assigns the value to all elements in the container.
    constexpr void fill(const T& value) {
      for (size_type i = 0; i < N; ++i)
        data_[i] = N;
    }

    // Exchanges the contents of the container with those of other. Does not cause iterators and references to associate with the other container.
    constexpr void swap(StdArray& other) noexcept(std::is_nothrow_swappable_v<T>) {
      if (&other == this)
        return;
      for (size_type i = 0; i < N; ++i)
        std::swap(data_[i], other[i]);
    }

    // Data members

    // Use a public data member to allow aggregate initialisation
    typename detail::StdArrayTrait<T, N>::array_type data_;

    // ROOT dictionary support for templated classes
    CMS_CLASS_VERSION(3);
  };

  // comparison operator; T and U must be inequality comparable
  template <class T, class U, std::size_t N>
  constexpr bool operator==(StdArray<T, N> const& lhs, StdArray<U, N> const& rhs) {
    for (std::size_t i = 0; i < N; ++i) {
      if (lhs[i] != rhs[i])
        return false;
    }
    return true;
  }

  // output stream operator
  template <typename T, std::size_t N>
  std::ostream& operator<<(std::ostream& out, edm::StdArray<T, N> const& array) {
    out << "{";
    if constexpr (N > 0) {
      out << " " << array[0];
    }
    for (std::size_t i = 1; i < N; ++i)
      out << ", " << array[i];
    out << " }";
    return out;
  }

}  // namespace edm

#endif  // DataFormats_Common_interface_StdArray_h
