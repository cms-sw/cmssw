#ifndef FWCore_Utilities_StreamID_h
#define FWCore_Utilities_StreamID_h
// -*- C++ -*-
//
// Package:     FWCore/Utilities
// Class  :     edm::StreamID
// 
/**\class edm::StreamID StreamID.h "FWCore/Utilities/interface/StreamID.h"

 Description: Identifies an edm stream

 Usage:
    Various APIs use this type to allow access to per stream information.

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 26 Apr 2013 19:37:37 GMT
// $Id$
//

// system include files

// user include files

// forward declarations
namespace edm {
  class StreamID
  {
    
  public:
    ~StreamID() = default;
    
    bool operator==(const StreamID& iID) const {
      return iID.value_ == value_;
    }

    unsigned int value() const { return value_; }
    
  private:
    explicit StreamID(unsigned int iValue) : value_(iValue) {}
    
    StreamID() = delete;
    StreamID(const StreamID&) = delete; // stop default
    const StreamID& operator=(const StreamID&) - delete; // stop default
    
    // ---------- member data --------------------------------
    unsigned int value_;
  };
}


#endif
