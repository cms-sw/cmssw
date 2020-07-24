#ifndef FWCore_Utilities_Span_h
#define FWCore_Utilities_Span_h

#include <cstddef>

namespace edm {
  /*
      *An edm::Span wraps begin() and end() iterators to a contiguous sequence
      of objects with the first element of the sequence at position zero,
      In other words the iterators should refer to random-access containers.

      To be replaced with std::Span in C++20.
      */

  template <class T>
  class Span {
  public:
    Span(T begin, T end) : begin_(begin), end_(end) {}

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
