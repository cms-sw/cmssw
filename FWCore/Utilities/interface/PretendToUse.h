#ifndef FWCore_Utilities_PretendToUse_h
#define FWCore_Utilities_PretendToUse_h

// $Id: PretendToUse.h,v 1.2 2007/09/20 20:11:07 wmtan Exp $
//

/// This header defines the function template pretendToUse; this can be useful in faking
/// out compilers that complain about unused variables.
//

template <typename T> inline void pretendToUse(T const&) { }

#endif
