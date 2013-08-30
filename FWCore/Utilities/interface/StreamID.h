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
//

// system include files

// user include files

// forward declarations
namespace edm {
  class Schedule;
  class EventProcessor;
  
  class StreamID
  {
    
  public:
    ~StreamID() = default;
    StreamID(const StreamID&) = default;
    StreamID& operator=(const StreamID&) = default;
    
    bool operator==(const StreamID& iID) const {
      return iID.value_ == value_;
    }

    operator unsigned int() const {return value_;}
    
    /** \return value ranging from 0 to one less than max number of streams.
     */
    unsigned int value() const { return value_; }
    
    static StreamID invalidStreamID() {
      return StreamID(0xFFFFFFFFU);
    }
    
  private:
    ///Only a Schedule is allowed to create one of these
    friend class Schedule;
    friend class EventProcessor;
    explicit StreamID(unsigned int iValue) : value_(iValue) {}
    
    StreamID() = delete;
    
    // ---------- member data --------------------------------
    unsigned int value_;
  };
}


#endif
