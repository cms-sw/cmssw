#ifndef HeterogeneousCore_SonicCore_SonicRetryActionBase
#define HeterogeneousCore_SonicCore_SonicRetryActionBase

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>
#include <string>

class SonicClientBase;

// Base class for retry actions
class SonicRetryActionBase {
public:
  SonicRetryActionBase(const edm::ParameterSet& conf, SonicClientBase& client);
  virtual ~SonicRetryActionBase() = default;

  // Polymorphic base held only via unique_ptr behind the factory; copying/moving through the
  // base pointer would slice derived objects, so both are explicitly disabled.
  SonicRetryActionBase(const SonicRetryActionBase&) = delete;
  SonicRetryActionBase& operator=(const SonicRetryActionBase&) = delete;
  SonicRetryActionBase(SonicRetryActionBase&&) = delete;
  SonicRetryActionBase& operator=(SonicRetryActionBase&&) = delete;

  bool shouldRetry() const { return shouldRetry_; }  // Getter for shouldRetry_

  virtual void retry() = 0;  // Pure virtual function for execution logic
  virtual void start() = 0;  // Pure virtual function for execution logic for initialization

protected:
  void eval();                // interface for calling evaluate in client
  void finish(bool success);  // interface for calling finish directly in client

  SonicClientBase& client_;
  bool shouldRetry_;  // Flag to track if further retries should happen
};

// Define the factory for creating retry actions
using RetryActionFactory =
    edmplugin::PluginFactory<SonicRetryActionBase*(const edm::ParameterSet&, SonicClientBase& client)>;

#endif

#define DEFINE_RETRY_ACTION(type) DEFINE_EDM_PLUGIN(RetryActionFactory, type, #type);
