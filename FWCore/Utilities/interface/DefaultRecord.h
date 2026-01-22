#ifndef FWCore_Utilities_DefaultRecord_h
#define FWCore_Utilities_DefaultRecord_h
// -*- C++ -*-
// Package:     FWCore/Utilities
// Class  :     DefaultRecord
/// Description: Special type used to indicate the default Record for a data type
// Usage:
// An ESGetToken can be created with DefaultRecord to indicate
// the default Record for the data type should be used.
//

namespace edm {
  struct DefaultRecord {};
}  // namespace edm
#endif
