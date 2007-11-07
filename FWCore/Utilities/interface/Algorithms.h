#ifndef FWCore_Utilities_Algorithms_h
#define FWCore_Utilities_Algorithms_h

#include <algorithm>

namespace edm {

  /// Function templates that provide wrappers for standard algorithms,
  /// avoiding some duplication
  /// and assuring that incommensurate iterators are not used.

  /// wrapper for std::for_each
  template <typename ForwardSequence, typename Func>
  inline
  Func
  for_all(ForwardSequence& s, Func f) {
    return std::for_each(s.begin(), s.end(), f);
  }

  /// wrappers for copy
  template <typename ForwardSequence, typename Func>
  inline
  Func
  copy_all(ForwardSequence& s, Func f) {
    return std::copy(s.begin(), s.end(), f);
  }

  /// wrappers for std::find
  template <typename ForwardSequence, typename Datum>
  inline
  typename ForwardSequence::const_iterator
  find_in_all(ForwardSequence const& s, Datum const& d) {
    return std::find(s.begin(), s.end(), d);
  }

  template <typename ForwardSequence, typename Datum>
  inline
  typename ForwardSequence::iterator
  find_in_all(ForwardSequence& s, Datum const& d) {
    return std::find(s.begin(), s.end(), d);
  }

  template <typename ForwardSequence, typename Datum>
  inline
  bool
  search_all(ForwardSequence const& s, Datum const& d) {
    return std::find(s.begin(), s.end(), d) != s.end();
  }

  /// wrapper for std::binary_search
  template <typename ForwardSequence, typename Datum>
  inline
  bool
  binary_search_all(ForwardSequence const& s, Datum const& d) {
    return std::binary_search(s.begin(), s.end(), d);
  }

  /// wrappers for std::sort
  template <typename RandomAccessSequence>
  inline
  void
  sort_all(RandomAccessSequence & s) {
    std::sort(s.begin(), s.end());
  }

  template <typename RandomAccessSequence, typename Predicate>
  inline
  void
  sort_all(RandomAccessSequence & s, Predicate p) {
    std::sort(s.begin(), s.end(), p);
  }
}
#endif
