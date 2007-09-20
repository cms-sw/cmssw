#ifndef FWCORE_UTILITIES_COPY_ALL_INC
#define FWCORE_UTILITIES_COPY_ALL_INC

#include <algorithm>

namespace edm
{

  /// copy_all is a function template that provides a simple wrapper
  /// for std::copy, avoiding some duplication and assuring that
  /// incommensurate iterators are not used.

  template <class ForwardSequence, class Func>
  inline
  Func
  copy_all(ForwardSequence& s, Func f)
  {
    return std::copy(s.begin(), s.end(), f);
  }

}
#endif
