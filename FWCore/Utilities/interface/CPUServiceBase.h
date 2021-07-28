#ifndef FWCore_Utilities_CPUServiceBase_h
#define FWCore_Utilities_CPUServiceBase_h
// -*- C++ -*-
//
// Package:     FWCore/Utilities
// Class  :     CPUServiceBase
//
/**\class CPUServiceBase CPUServiceBase.h "CPUServiceBase.h"

 Description: Base class for CPU Services

 Usage:
    Provides an interface to allow us to query the existing
    CPU service.

*/
//
// Original Author:  Brian Bockelman
//         Created:  Wed Sep  7 12:05:13 CDT 2016
//

// system include files
#include <string>

// forward declarations
namespace edm {
  class CPUServiceBase {
  public:
    CPUServiceBase();
    CPUServiceBase(const CPUServiceBase &) = delete;                   // stop default
    const CPUServiceBase &operator=(const CPUServiceBase &) = delete;  // stop default

    virtual ~CPUServiceBase();

    // ---------- member functions ---------------------------
    ///CPU information - the models present and average speed.
    virtual bool cpuInfo(std::string &models, double &avgSpeed) = 0;
  };
}  // namespace edm

#endif
