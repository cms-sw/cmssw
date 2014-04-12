#ifndef FWCore_Utilities_CallNTimesNoWait_h
#define FWCore_Utilities_CallNTimesNoWait_h
// -*- C++ -*-
//
// Package:     FWCore/Utilities
// Class  :     CallNTimesNoWait
// 
/**\class edm::CallNTimesNoWait CallNTimesNoWait.h "CallNTimesNoWait.h"

 Description: Thread safe way to do something N times

 Usage:
    This class allows one to safely do a 'non-side-effect' operation N times in a job.
 An example use would be
 \code
    void myFunc( int iValue) {
      static CallNTimesNoWait message{2};
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
//         Created:  Fri, 15 Nov 2013 14:29:41 GMT
//

// system include files
#include<atomic>

// user include files

// forward declarations
namespace edm {
  class CallNTimesNoWait
  {
    
  public:
    CallNTimesNoWait( unsigned short iNTimes ): m_ntimes(static_cast<int>(iNTimes)-1), m_done(false){}
    
    template <typename T>
    void operator()(T iCall) {
      if(not m_done.load(std::memory_order_acquire) ) {
        if(m_ntimes.fetch_sub(1,std::memory_order_acq_rel)<0) {
          m_done.store(true,std::memory_order_release);
          return;
        };
        iCall();
      }
    }
    
	private:
    std::atomic<int> m_ntimes;
    std::atomic<bool> m_done;
  };
}

#endif
