#ifndef HeterogeneousCore_SonicTriton_RetryFallbackServerAction_h
#define HeterogeneousCore_SonicTriton_RetryFallbackServerAction_h

#include "HeterogeneousCore/SonicCore/interface/RetryActionBase.h"

class RetryFallbackServerAction : public RetryActionBase {
public:
  RetryFallbackServerAction(const edm::ParameterSet& conf, SonicClientBase* client);
  ~RetryFallbackServerAction() override = default;

  void retry() override;
  void start() override;

private:
  unsigned tries_;
};

#endif
