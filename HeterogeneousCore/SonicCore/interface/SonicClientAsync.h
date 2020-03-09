#ifndef HeterogeneousCore_SonicCore_SonicClientAsync
#define HeterogeneousCore_SonicCore_SonicClientAsync

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"

#include "HeterogeneousCore/SonicCore/interface/SonicClientBase.h"
#include "HeterogeneousCore/SonicCore/interface/SonicClientTypes.h"

template <typename InputT, typename OutputT = InputT>
class SonicClientAsync : public SonicClientBase, public SonicClientTypes<InputT, OutputT> {
public:
  virtual ~SonicClientAsync() {}

  //main operation
  void dispatch(edm::WaitingTaskWithArenaHolder holder) override final {
    holder_ = std::move(holder);
    setStartTime();
    evaluate();
    //impl calls finish() which calls holder_
  }
};

#endif
