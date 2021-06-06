#include "HeterogeneousCore/SonicCore/interface/SonicDispatcherPseudoAsync.h"
#include "HeterogeneousCore/SonicCore/interface/SonicClientBase.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

SonicDispatcherPseudoAsync::SonicDispatcherPseudoAsync(SonicClientBase* client)
    : SonicDispatcher(client), hasCall_(false), stop_(false) {
  thread_ = std::make_unique<std::thread>([this]() { waitForNext(); });
}

SonicDispatcherPseudoAsync::~SonicDispatcherPseudoAsync() {
  stop_ = true;
  cond_.notify_one();
  if (thread_) {
    // avoid throwing in destructor
    CMS_SA_ALLOW try {
      thread_->join();
      thread_.reset();
    } catch (...) {
    }
  }
}

void SonicDispatcherPseudoAsync::dispatch(edm::WaitingTaskWithArenaHolder holder) {
  //do all read/writes inside lock to ensure cache synchronization
  {
    std::lock_guard<std::mutex> guard(mutex_);
    client_->start(std::move(holder));

    //activate thread to wait for response, and return
    hasCall_ = true;
  }
  cond_.notify_one();
}

void SonicDispatcherPseudoAsync::waitForNext() {
  while (true) {
    //wait for condition
    {
      std::unique_lock<std::mutex> lk(mutex_);
      cond_.wait(lk, [this]() { return (hasCall_ or stop_); });
      if (stop_)
        break;

      //do everything inside lock
      client_->evaluate();

      //reset condition
      hasCall_ = false;
    }
  }
}
