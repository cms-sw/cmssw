#ifndef FWCore_Utilities_RunIndex_h
#define FWCore_Utilities_RunIndex_h
// -*- C++ -*-
//
// Package:     FWCore/Utilities
// Class  :     edm::RunIndex
// 
/**\class edm::RunIndex RunIndex.h "FWCore/Utilities/interface/RunIndex.h"

 Description: Identifies a 'slot' being used to hold an active Run

 Usage:
    Various APIs used this to access per Run information.
 It is important to realize that the same RunIndex may be used to refer
 to different Runs over the lifetime of a job. An RunIndex will only get
 a new Run after the previous Run using that index has finished being used.

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 26 Apr 2013 19:38:56 GMT
// $Id$
//

// system include files

// user include files

// forward declarations
namespace edm {
  class RunIndex
  {
    
  public:
    ~RunIndex() = default;
    
    // ---------- const member functions ---------------------
    bool operator==(const RunIndex& iIndex) const {
      return value() == iIndex.value();
    }
    unsigned int value() const { return value_;}
    
    
  private:
    explicit RunIndex(unsigned int iIndex) : value_(iIndex) {}

    RunIndex() = delete;
    RunIndex(const RunIndex&) = delete; // stop default
    const RunIndex& operator=(const RunIndex&) = delete; // stop default
    
    // ---------- member data --------------------------------
    unsigned int value_;
    
  };
}


#endif
