#ifndef FWCore_Concurrency_WaitingThreadPool_h
#define FWCore_Concurrency_WaitingThreadPool_h

#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"

#include <condition_variable>
#include <mutex>
#include <thread>

namespace edm {
  namespace impl {
    class WaitingThread {
    public:
      WaitingThread();
      ~WaitingThread() noexcept;

      WaitingThread(WaitingThread const&) = delete;
      WaitingThread& operator=(WaitingThread&&) = delete;
      WaitingThread(WaitingThread&&) = delete;
      WaitingThread& operator=(WaitingThread const&) = delete;

      template <typename F>
      void run(WaitingTaskWithArenaHolder holder, F&& func, std::shared_ptr<WaitingThread> thisPtr) {
        std::unique_lock lk(mutex_);
        func_ = [holder = std::move(holder), func = std::forward<F>(func)]() mutable {
          CMS_SA_ALLOW try { func(); } catch (...) {
            holder.doneWaiting(std::current_exception());
          }
        };
        thisPtr_ = std::move(thisPtr);
        cond_.notify_one();
      }

    private:
      void stopThread() {
        std::unique_lock lk(mutex_);
        stopThread_ = true;
        cond_.notify_one();
      }

      void threadLoop() noexcept;

      std::thread thread_;
      std::mutex mutex_;
      std::condition_variable cond_;
      CMS_THREAD_GUARD(mutex_) std::function<void()> func_;
      // The purpose of thisPtr_ is to keep the WaitingThread object
      // outside of the WaitingThreadPool until the func_ has returned.
      CMS_THREAD_GUARD(mutex_) std::shared_ptr<WaitingThread> thisPtr_;
      CMS_THREAD_GUARD(mutex_) bool stopThread_ = false;
    };
  }  // namespace impl

  // Provides a mechanism to run the function 'func' asynchronously,
  // i.e. without the calling thread to wait for the func() to return.
  // The func should do as little work (outside of the TBB threadpool)
  // as possible. The func must terminate eventually. The intended use
  // case are blocking synchronization calls with external entities,
  // where the calling thread is suspended while waiting.
  //
  // The func() is run in a thread that belongs to a separate pool of
  // threads than the calling thread. Remotely similar to
  // std::async(), but instead of dealing with std::futures, takes an
  // edm::WaitingTaskWithArenaHolder object, that is signaled upon the
  // func() returning or throwing an exception.
  class WaitingThreadPool {
  public:
    WaitingThreadPool() = default;
    WaitingThreadPool(WaitingThreadPool const&) = delete;
    WaitingThreadPool& operator=(WaitingThreadPool const&) = delete;
    WaitingThreadPool(WaitingThreadPool&&) = delete;
    WaitingThreadPool& operator=(WaitingThreadPool&&) = delete;

    template <typename F>
    void runAsync(WaitingTaskWithArenaHolder holder, F&& func) {
      auto thread = pool_.makeOrGet([]() { return std::make_unique<impl::WaitingThread>(); });
      thread->run(std::move(holder), std::forward<F>(func), std::move(thread));
    }

  private:
    edm::ReusableObjectHolder<impl::WaitingThread> pool_;
  };
}  // namespace edm

#endif
