#ifndef HeterogeneousCore_SonicCore_SonicClientSync
#define HeterogeneousCore_SonicCore_SonicClientSync

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"

#include "HeterogeneousCore/SonicCore/interface/SonicClientBase.h"
#include "HeterogeneousCore/SonicCore/interface/SonicClientTypes.h"

#include <exception>

template <typename InputT, typename OutputT = InputT>
class SonicClientSync : public SonicClientBase, public SonicClientTypes<InputT, OutputT> {
public:
  //main operation
  void dispatch(edm::WaitingTaskWithArenaHolder holder) final {
    holder_ = std::move(holder);
    setStartTime();

    evaluate();
  }
};

#endif
