#include "FWCore/Concurrency/interface/ThreadSafeOutputFileStream.h"

#include <sstream>

namespace edm {

  ThreadSafeOutputFileStream::ThreadSafeOutputFileStream(std::string const& name)
    : file_{name}
  {}

  ThreadSafeOutputFileStream::~ThreadSafeOutputFileStream()
  {
    std::string msg;
    while (waitingMessages_.try_pop(msg)) {
      file_ << msg << '\n';
    }
    file_.close();
  }

  void
  ThreadSafeOutputFileStream::operator<<(std::string msg)
  {
    bool expected {false};
    if(msgBeingLogged_.compare_exchange_strong(expected, true)) {
      do {
        file_ << msg << '\n';
      } while (waitingMessages_.try_pop(msg));
      msgBeingLogged_.store(false);
    } else {
      waitingMessages_.push(std::move(msg));
    }
  }
}
