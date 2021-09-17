#ifndef HeterogeneousCore_SonicCore_SonicDispatcherPseudoAsync
#define HeterogeneousCore_SonicCore_SonicDispatcherPseudoAsync

#include "HeterogeneousCore/SonicCore/interface/SonicDispatcher.h"

#include <memory>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <atomic>
#include <exception>

class SonicClientBase;

//pretend to be async + non-blocking by waiting for blocking calls to return in separate std::thread
class SonicDispatcherPseudoAsync : public SonicDispatcher {
public:
  //constructor
  SonicDispatcherPseudoAsync(SonicClientBase* client);

  //destructor
  ~SonicDispatcherPseudoAsync() override;

  //main operation
  void dispatch(edm::WaitingTaskWithArenaHolder holder) override;

private:
  void waitForNext();

  //members
  bool hasCall_;
  std::mutex mutex_;
  std::condition_variable cond_;
  std::atomic<bool> stop_;
  std::unique_ptr<std::thread> thread_;
};

#endif
