#ifndef Framework_Utilities_make_sentry_h
#define Framework_Utilities_make_sentry_h
// -*- C++ -*-
//
// Package:     Framework/Utilities
// Class  :     make_sentry
// 
/**\function make_sentry make_sentry.h "FWCore/Utilities/interface/make_sentry.h"

 Description: Creates a std::unique_ptr from a lambda to be used as a sentry

 Usage:
    <usage>

*/
//
// Original Author:  root
//         Created:  Fri, 19 Aug 2016 20:02:12 GMT
//

// system include files
#include <memory>
// user include files

// forward declarations

namespace edm {
  ///NOTE: if iObject is null, then iFunc will not be called
  template<typename T, typename F> std::unique_ptr<T, F> make_sentry(T* iObject, F iFunc) {
    return std::unique_ptr<T, F>(iObject, iFunc);
  }
}

#endif
