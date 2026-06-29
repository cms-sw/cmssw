#ifndef FWCore_Concurrency_SpinLock_h
#define FWCore_Concurrency_SpinLock_h
//
// Package:     FWCore/Concurrency
// Class  :     SpinLock
//
/**\class edm::SpinLock SpinLock.h "FWCore/Concurrency/interface/SpinLock.h"

 Description: A trivial lock which continues looping until the lock is acquired

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Tue, 23 Jun 2026 14:12:13 GMT
//

// system include files
#include <atomic>

// user include files
#include "FWCore/Concurrency/interface/hardware_pause.h"

// forward declarations

namespace edm {
  class SpinLock {
  public:
    SpinLock() noexcept = default;
    ~SpinLock() noexcept {}

    SpinLock(const SpinLock&) = delete;
    SpinLock& operator=(const SpinLock&) = delete;
    SpinLock(SpinLock&&) = delete;
    SpinLock& operator=(SpinLock&&) = delete;

    // ---------- const member functions ---------------------

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------
    void lock() noexcept {
      //we acquire if the previous value is false
      while (locked_.exchange(true, std::memory_order_acq_rel)) {
        hardware_pause();
      }
    }
    void unlock() noexcept {
      locked_.store(false, std::memory_order_release);
      ;
    }

  private:
    // ---------- member data --------------------------------
    std::atomic<bool> locked_{false};
  };
}  // namespace edm
#endif
