#ifndef FWCORE_UTILITIES_FOR_ALL_INC
#define FWCORE_UTILITIES_FOR_ALL_INC

#include <algorithm>

namespace edm
{

  /// for_all is a function template that provides a simple wrapper
  /// for std::for_each, avoiding some duplication and assuring that
  /// incommensurate iterators are not used.

  template <class ForwardSequence, class Func>
  inline
  Func
  for_all(ForwardSequence& s, Func f)
  {
    return std::for_each(s.begin(), s.end(), f);
  }

}
#endif
