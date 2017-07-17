#ifndef __FWCore_Utilities_interface_csa_assert_h__
#define __FWCore_Utilities_interface_csa_assert_h__

#include <cassert>

#ifdef __clang_analyzer__
#define csa_assert(CONDITION) assert(CONDITION)
#else
#define csa_assert(CONDITION) (void)0
#endif

#endif // __FWCore_Utilities_interface_csa_assert_h__
