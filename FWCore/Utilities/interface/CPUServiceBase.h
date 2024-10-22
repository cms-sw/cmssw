#ifndef FWCore_Utilities_CPUServiceBase_h
#define FWCore_Utilities_CPUServiceBase_h
// -*- C++ -*-
//
// Package:     FWCore/Utilities
// Class  :     CPUServiceBase
//
/**\class edm::CPUServiceBase

 Description: Base class for CPU Services

 Usage:
    Provides an interface to allow us to query the existing
    CPU service.

*/
//
// Original Author:  Brian Bockelman
//         Created:  Wed Sep  7 12:05:13 CDT 2016
//

namespace edm {
  class CPUServiceBase {
  public:
    CPUServiceBase();
    CPUServiceBase(const CPUServiceBase &) = delete;                   // stop default
    const CPUServiceBase &operator=(const CPUServiceBase &) = delete;  // stop default

    virtual ~CPUServiceBase();
  };
}  // namespace edm

#endif
