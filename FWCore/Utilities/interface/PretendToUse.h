#ifndef FWCore_Utilities_PretendToUse_h
#define FWCore_Utilities_PretendToUse_h

//

/// This header defines the function template pretendToUse; this can be useful in faking
/// out compilers that complain about unused variables.
//

template <typename T> inline void pretendToUse(T const&) { }

#endif
