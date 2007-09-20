#ifndef FWCore_Utilities_copy_all_h
#define FWCore_Utilities_copy_all_h

#include <algorithm>

namespace edm
{

  /// copy_all is a function template that provides a simple wrapper
  /// for std::copy, avoiding some duplication and assuring that
  /// incommensurate iterators are not used.

  template <typename ForwardSequence, typename Func>
  inline
  Func
  copy_all(ForwardSequence& s, Func f)
  {
    return std::copy(s.begin(), s.end(), f);
  }

}
#endif
