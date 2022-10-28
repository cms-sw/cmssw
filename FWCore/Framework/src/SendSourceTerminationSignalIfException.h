#ifndef FWCore_Framework_SendSourceTerminationSignalIfException_h
#define FWCore_Framework_SendSourceTerminationSignalIfException_h
//
// Package:     FWCore/Framework
// Class  :     SendSourceTerminationSignalIfException
//
/**\class edm::SendSourceTerminationSignalIfException
*/

#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/TerminationOrigin.h"

namespace edm {

  //Sentry class to only send a signal if an
  // exception occurs. An exception is identified
  // by the destructor being called without first
  // calling completedSuccessfully().
  class SendSourceTerminationSignalIfException {
  public:
    SendSourceTerminationSignalIfException(ActivityRegistry* iReg) : reg_(iReg) {}
    ~SendSourceTerminationSignalIfException() {
      if (reg_) {
        reg_->preSourceEarlyTerminationSignal_(TerminationOrigin::ExceptionFromThisContext);
      }
    }
    void completedSuccessfully() { reg_ = nullptr; }

  private:
    ActivityRegistry* reg_;  // We do not use propagate_const because the registry itself is mutable.
  };
}  // namespace edm

#endif
