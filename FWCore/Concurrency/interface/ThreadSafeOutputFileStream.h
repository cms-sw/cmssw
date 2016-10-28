#ifndef FWCore_Concurrency_ThreadSafeOutputFileStream_h
#define FWCore_Concurrency_ThreadSafeOutputFileStream_h

#include "tbb/concurrent_queue.h"

#include <atomic>
#include <fstream>
#include <string>

namespace edm {
  class ThreadSafeOutputFileStream {
  public:
    ThreadSafeOutputFileStream(std::string const& name);
    ~ThreadSafeOutputFileStream();

    // Return type of operator<< is 'void' to discourage chaining,
    // which is inefficient for this implementation.
    void operator<<(std::string msg);
  private:
    std::ofstream file_;
    std::atomic<bool> msgBeingLogged_ {false};
    tbb::concurrent_queue<std::string> waitingMessages_ {};
  };
}

#endif
