#ifndef HeterogeneousCore_SonicCore_SonicClientSync
#define HeterogeneousCore_SonicCore_SonicClientSync

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"

#include "HeterogeneousCore/SonicCore/interface/SonicClientBase.h"
#include "HeterogeneousCore/SonicCore/interface/SonicClientTypes.h"

#include <exception>

template <typename InputT, typename OutputT = InputT>
class SonicClientSync : public SonicClientBase, public SonicClientTypes<InputT, OutputT> {
public:
  virtual ~SonicClientSync() {}

  //main operation
  void dispatch(edm::WaitingTaskWithArenaHolder holder) override final {
    holder_ = std::move(holder);
    setStartTime();

    std::exception_ptr eptr;
    try {
      evaluate();
    } catch (...) {
      eptr = std::current_exception();
    }

    //sync Client calls holder at the end
    finish(eptr);
  }
};

#endif
