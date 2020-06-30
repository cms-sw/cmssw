#ifndef FWCore_Utilities_Range_h
#define FWCore_Utilities_Range_h

#include <cstddef>

namespace edm {
  /*
      *class which implements begin() and end() to use range-based loop with
      pairs of iterators or pointers.
      */

  template <class T>
  class Range {
  public:
    Range(T begin, T end) : begin_(begin), end_(end) {}

    T begin() const { return begin_; }
    T end() const { return end_; }

    bool empty() const { return begin_ == end_; }
    auto size() const { return end_ - begin_; }

    auto const& operator[](std::size_t idx) const { return *(begin_ + idx); }

    auto const& front() const { return *begin_; }
    auto const& back() const { return *(end_ - 1); }

  private:
    const T begin_;
    const T end_;
  };
};  // namespace edm

#endif
