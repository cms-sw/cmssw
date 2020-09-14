#ifndef HeterogeneousCore_SonicCore_SonicDispatcher
#define HeterogeneousCore_SonicCore_SonicDispatcher

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"

class SonicClientBase;

class SonicDispatcher {
public:
  //constructor
  SonicDispatcher(SonicClientBase* client) : client_(client) {}

  //destructor
  virtual ~SonicDispatcher() = default;

  //main operation
  virtual void dispatch(edm::WaitingTaskWithArenaHolder holder);

protected:
  SonicClientBase* client_;
};

#endif
