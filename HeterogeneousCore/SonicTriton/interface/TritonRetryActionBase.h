#ifndef HeterogeneousCore_SonicTriton_TritonRetryActionBase
#define HeterogeneousCore_SonicTriton_TritonRetryActionBase

#include "HeterogeneousCore/SonicCore/interface/SonicRetryActionBase.h"

class TritonClient;

// Common base for Triton-specific retry actions: provides tritonClient(), a shared
// SonicClientBase& -> TritonClient& downcast with a clear error on mismatch, so every
// Triton retry action checks the client type the same way.
class TritonRetryActionBase : public SonicRetryActionBase {
public:
  TritonRetryActionBase(const edm::ParameterSet& conf, SonicClientBase& client) : SonicRetryActionBase(conf, client) {}

protected:
  // Called lazily from retry(), not the constructor -- see TritonRetryActionBase.cc.
  TritonClient& tritonClient() const;
};

#endif
