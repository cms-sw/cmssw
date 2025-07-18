#ifndef FWCore_Concurrency_Async_h
#define FWCore_Concurrency_Async_h

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Concurrency/interface/WaitingThreadPool.h"

namespace edm {
  // All member functions are thread safe
  class Async {
  public:
    Async() = default;
    virtual ~Async() noexcept;

    // prevent copying and moving
    Async(Async const&) = delete;
    Async(Async&&) = delete;
    Async& operator=(Async const&) = delete;
    Async& operator=(Async&&) = delete;

    template <typename F, typename G>
    void runAsync(WaitingTaskWithArenaHolder holder, F&& func, G&& errorContextFunc) {
      ensureAllowed();
      pool_.runAsync(std::move(holder), std::forward<F>(func), std::forward<G>(errorContextFunc));
    }

  protected:
    virtual void ensureAllowed() const = 0;

  private:
    WaitingThreadPool pool_;
  };
}  // namespace edm

#endif
