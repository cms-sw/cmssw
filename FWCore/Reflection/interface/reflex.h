#ifndef FWCore_Reflection_interface_reflex_h
#define FWCore_Reflection_interface_reflex_h

/* These macros can be used to annotate the data members of DataFormats classes
 * and achieve the same effect of ROOT-style comments or reflex dictionaries:
 * 
 * private:
 *   int size_;
 *   float* data_;       //[size_]
 *   float* transient_;  //!
 * 
 * can be expressed as
 * 
 * private:
 *   int size_;
 *   float* data_ EDM_REFLEX_SIZE(size_);
 *   float* transient_ EDM_REFLEX_TRANSIENT;
 * 
 * The main advantage is that - unlike comments - these macros can be used inside
 * other macros.
 *
 * To avoid warning about unrecognised attributes, these macros expand to nothing
 * unless __CLING__ is defined.
 */

#include "FWCore/Utilities/interface/stringize.h"

#ifdef __CLING__

// Macros used to annotate class members for the generation of ROOT dictionaries
#define EDM_REFLEX_TRANSIENT [[clang::annotate("!")]]
#define EDM_REFLEX_SIZE(SIZE) [[clang::annotate("[" EDM_STRINGIZE(SIZE) "]")]]

#else

#define EDM_REFLEX_TRANSIENT
#define EDM_REFLEX_SIZE(SIZE)

#endif  // __CLING__

#endif  // FWCore_Reflection_interface_reflex_h
