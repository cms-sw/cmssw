#ifndef FWCORE_UTILITIES_PRETEND_TO_USE_H
#define FWCORE_UTILITIES_PRETEND_TO_USE_H

// $Id:$
//

/// This header defines the function template pretendToUse; this can be useful in faking
/// out compilers that complain about unused variables.
//

template <class T> inline void pretendToUse(T const&) { }

#endif
