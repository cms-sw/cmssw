#ifndef FWCore_Utilities_VecArray_h
#define FWCore_Utilities_VecArray_h

#include <utility>
#include <stdexcept>
#include <string>
#include <cstddef>

namespace edm {
  /**
   * A class for extending std::array with std::vector-like interface.
   *
   * This class can be useful if the maximum length is known at
   * compile-time (can use std::array), and that the length is rather
   * small (maximum size of std::array is comparable with the overhead
   * of std::vector). It is also free of dynamic memory allocations.
   *
   * Note that the implemented interface is not complete compared to
   * std::array or std:vector. Feel free contribute if further needs
   * arise.
   *
   * The second template argument is unsigned int (not size_t) on
   * purpose to reduce the size of the class by 4 bytes (at least in
   * Linux amd64). For all practical purposes even unsigned char could
   * be enough.
   */
  template <typename T, unsigned int N>
  class VecArray {
    T data_[N];
    unsigned int size_;

  public:
    using value_type = T;
    using size_type = unsigned int;
    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using iterator = value_type*;
    using const_iterator = const value_type*;

    VecArray() : data_{}, size_{0} {}

    // Not range-checked, undefined behaviour if access beyond size()
    reference operator[](size_type pos) { return data_[pos]; }
    // Not range-checked, undefined behaviour if access beyond size()
    const_reference operator[](size_type pos) const { return data_[pos]; }
    // Undefined behaviour if size()==0
    reference front() { return data_[0]; }
    // Undefined behaviour if size()==0
    const_reference front() const { return data_[0]; }

    // Undefined behaviour if size()==0
    reference back() { return data_[size_ - 1]; }
    // Undefined behaviour if size()==0
    const_reference back() const { return data_[size_ - 1]; }
    pointer data() { return data_; }
    const_pointer data() const { return data_; }

    iterator begin() noexcept { return data_; }
    const_iterator begin() const noexcept { return data_; }
    const_iterator cbegin() const noexcept { return data_; }

    iterator end() noexcept { return begin() + size_; }
    const_iterator end() const noexcept { return begin() + size_; }
    const_iterator cend() const noexcept { return cbegin() + size_; }

    constexpr bool empty() const noexcept { return size_ == 0; }
    constexpr size_type size() const noexcept { return size_; }
    static constexpr size_type capacity() noexcept { return N; }

    void clear() { size_ = 0; }

    // Throws if size()==N
    void push_back(const T& value) {
      if (size_ >= N)
        throw std::length_error("push_back on already-full VecArray (N=" + std::to_string(N) + ")");
      push_back_unchecked(value);
    }

    // Undefined behaviour if size()==N
    void push_back_unchecked(const T& value) {
      data_[size_] = value;
      ++size_;
    }

    // Throws if size()==N
    template <typename... Args>
    void emplace_back(Args&&... args) {
      if (size_ >= N)
        throw std::length_error("emplace_back on already-full VecArray (N=" + std::to_string(N) + ")");
      emplace_back_unchecked(std::forward<Args>(args)...);
    }

    // Undefined behaviour if size()==N
    template <typename... Args>
    void emplace_back_unchecked(Args&&... args) {
      data_[size_] = T(std::forward<Args>(args)...);
      ++size_;
    }

    // Undefined behaviour if size()==0
    void pop_back() { --size_; }

    void resize(unsigned int size) {
      if (size > N)
        throw std::length_error("Requesting size " + std::to_string(size) + " while maximum allowed is " +
                                std::to_string(N));

      while (size < size_)
        pop_back();
      size_ = size;
    }

    void swap(VecArray& other) noexcept(noexcept(std::swap(data_, other.data_)) && noexcept(std::swap(size_,
                                                                                                      other.size_))) {
      std::swap(data_, other.data_);
      std::swap(size_, other.size_);
    }
  };
}  // namespace edm

#endif
