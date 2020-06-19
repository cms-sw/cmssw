#ifndef HeterogeneousCore_SonicCore_SonicClientBase
#define HeterogeneousCore_SonicCore_SonicClientBase

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <chrono>
#include <exception>

class SonicClientBase {
public:
  //constructor
  SonicClientBase() : tries_(0) {}

  //destructor
  virtual ~SonicClientBase() = default;

  void setDebugName(const std::string& debugName) {
    debugName_ = debugName;
    fullDebugName_ = debugName_;
    if (!clientName_.empty())
      fullDebugName_ += ":" + clientName_;
  }
  const std::string& debugName() const { return debugName_; }
  const std::string& clientName() const { return clientName_; }

  //main operation
  virtual void dispatch(edm::WaitingTaskWithArenaHolder holder) = 0;

protected:
  virtual void evaluate() = 0;

  //this should be overridden by clients that allow retries
  virtual unsigned allowedTries() const { return 0; }

  void setStartTime() {
    tries_ = 0;
    if (debugName_.empty())
      return;
    t0_ = std::chrono::high_resolution_clock::now();
  }

  void finish(bool success, std::exception_ptr eptr = std::exception_ptr{}) {
    //retries are only allowed if no exception was raised
    if (!success and !eptr) {
      ++tries_;
      //if max retries has not been exceeded, call evaluate again
      if (tries_ < allowedTries()) {
        evaluate();
        //avoid calling doneWaiting() twice
        return;
      }
      //prepare an exception if exceeded
      else {
        cms::Exception ex("SonicCallFailed");
        ex << "call failed after max " << tries_ << " tries";
        eptr = make_exception_ptr(ex);
      }
    }
    if (!debugName_.empty()) {
      auto t1 = std::chrono::high_resolution_clock::now();
      edm::LogInfo(fullDebugName_) << "Client time: "
                                   << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0_).count();
    }
    holder_.doneWaiting(eptr);
  }

  //members
  unsigned tries_;
  edm::WaitingTaskWithArenaHolder holder_;

  //for logging/debugging
  std::string clientName_, debugName_, fullDebugName_;
  std::chrono::time_point<std::chrono::high_resolution_clock> t0_;
};

#endif
