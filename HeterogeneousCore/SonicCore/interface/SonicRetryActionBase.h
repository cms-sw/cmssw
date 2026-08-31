#ifndef HeterogeneousCore_SonicCore_SonicRetryActionBase
#define HeterogeneousCore_SonicCore_SonicRetryActionBase

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/SonicCore/interface/SonicClientBase.h"
#include <memory>
#include <string>

// Base class for retry actions
class SonicRetryActionBase {
public:
  SonicRetryActionBase(const edm::ParameterSet& conf, SonicClientBase* client);
  virtual ~SonicRetryActionBase() = default;

  bool shouldRetry() const { return shouldRetry_; }  // Getter for shouldRetry_

  virtual void retry() = 0;  // Pure virtual function for execution logic
  virtual void start() = 0;  // Pure virtual function for execution logic for initialization

protected:
  void eval();                // interface for calling evaluate in client
  void finish(bool success);  // interface for calling finish directly in client

  SonicClientBase* client_;
  bool shouldRetry_;  // Flag to track if further retries should happen
};

// Define the factory for creating retry actions
using RetryActionFactory =
    edmplugin::PluginFactory<SonicRetryActionBase*(const edm::ParameterSet&, SonicClientBase* client)>;

#endif

#define DEFINE_RETRY_ACTION(type) DEFINE_EDM_PLUGIN(RetryActionFactory, type, #type);
