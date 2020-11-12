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

  //alternate operation when ExternalWork is not used
  virtual void dispatch();

protected:
  SonicClientBase* client_;
};

#endif
