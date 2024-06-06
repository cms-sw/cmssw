#include "FWCore/Concurrency/interface/WaitingThreadPool.h"

#include <cassert>
#include <string_view>

#include <pthread.h>

namespace edm::impl {
  WaitingThread::WaitingThread() {
    thread_ = std::thread(&WaitingThread::threadLoop, this);
    static constexpr auto poolName = "edm async pool";
    // pthread_setname_np() string length is limited to 16 characters,
    // including the null termination
    static_assert(std::string_view(poolName).size() < 16);

    int err = pthread_setname_np(thread_.native_handle(), poolName);
    // According to the glibc documentation, the only error
    // pthread_setname_np() can return is about the argument C-string
    // being too long. We already check above the C-string is shorter
    // than the limit was at the time of writing. In order to capture
    // if the limit shortens, or other error conditions get added,
    // let's assert() anyway (exception feels overkill)
    assert(err == 0);
  }

  WaitingThread::~WaitingThread() noexcept {
    // When we are shutting down, we don't care about any possible
    // system errors anymore
    CMS_SA_ALLOW try {
      stopThread();
      thread_.join();
    } catch (...) {
    }
  }

  void WaitingThread::threadLoop() noexcept {
    std::unique_lock lk(mutex_);

    while (true) {
      cond_.wait(lk, [this]() { return static_cast<bool>(func_) or stopThread_; });
      if (stopThread_) {
        // There should be no way to stop the thread when it as the
        // func_ assigned, but let's make sure
        assert(not thisPtr_);
        break;
      }
      func_();
      // Must return this WaitingThread to the ReusableObjectHolder in
      // the WaitingThreadPool before resettting func_ (that holds the
      // WaitingTaskWithArenaHolder, that enables the progress in the
      // TBB thread pool) in order to meet the requirement of
      // ReusableObjectHolder destructor that there are no outstanding
      // objects.
      thisPtr_.reset();
      decltype(func_)().swap(func_);
    }
  }
}  // namespace edm::impl
