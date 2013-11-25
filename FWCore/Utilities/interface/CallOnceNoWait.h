#ifndef FWCore_Utilities_CallOnceNoWait_h
#define FWCore_Utilities_CallOnceNoWait_h
// -*- C++ -*-
//
// Package:     FWCore/Utilities
// Class  :     CallOnceNoWait
// 
/**\class edm::CallOnceNoWait CallOnceNoWait.h "FWCore/Utilities/interface/CallOnceNoWait.h"

 Description: Thread safe way to do something 1 time

 Usage:
 This class allows one to safely do a 'non-side-effect' operation 1 time in a job.
 An example use would be
 \code
 void myFunc( int iValue) {
 static CallOnceNoWait message;
 message([&]() { edm::LogInfo("IWantToKnow")<<"called with "<<iValue; } );
 \endcode
 The important thing to remember, is there is no guarantee that the operation being run
 finishes before a thread which doesn't get to run the operation reaches the code following
 the call. Therefore it is useful to suppress messages but should not be used to do something
 like filling a container with values since the filling is not guaranteed to complete before
 another thread skips the call.

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 15 Nov 2013 14:29:51 GMT
//

// system include files
#include <atomic>

// user include files

// forward declarations

namespace edm {
  class CallOnceNoWait
  {
  public:
    CallOnceNoWait() : m_called(false) {}
    
    template <typename T>
    void operator()(T iCall) {
      bool expected = false;
      if(m_called.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        iCall();
      }
    }
    
  private:
    std::atomic<bool> m_called;
  };
}


#endif
