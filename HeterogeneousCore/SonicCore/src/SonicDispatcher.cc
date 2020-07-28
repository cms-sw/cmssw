#include "HeterogeneousCore/SonicCore/interface/SonicDispatcher.h"
#include "HeterogeneousCore/SonicCore/interface/SonicClientBase.h"

void SonicDispatcher::dispatch(edm::WaitingTaskWithArenaHolder holder) {
  client_->start(std::move(holder));
  client_->evaluate();
}
