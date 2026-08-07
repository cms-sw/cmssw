#ifndef HeterogeneousCore_SonicCore_SonicClientBase
#define HeterogeneousCore_SonicCore_SonicClientBase

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HeterogeneousCore/SonicCore/interface/SonicDispatcher.h"
#include "HeterogeneousCore/SonicCore/interface/SonicDispatcherPseudoAsync.h"

#include <string>
#include <vector>
#include <exception>
#include <memory>
#include <optional>

enum class SonicMode { Sync = 1, Async = 2, PseudoAsync = 3 };

class RetryActionBase;

class SonicClientBase {
public:
  //constructor
  SonicClientBase(const edm::ParameterSet& params, const std::string& debugName, const std::string& clientName);

  //destructor
  virtual ~SonicClientBase() = default;

  const std::string& debugName() const { return debugName_; }
  const std::string& clientName() const { return clientName_; }
  SonicMode mode() const { return mode_; }

  //main operation
  virtual void dispatch(edm::WaitingTaskWithArenaHolder holder) { dispatcher_->dispatch(std::move(holder)); }

  //alternate operation when ExternalWork is not used
  virtual void dispatch() { dispatcher_->dispatch(); }

  //helper: does nothing by default
  virtual void reset() {}

  //provide base params
  //defaultRetryType: retryType used for the default entry of the "Retry" VPSet.
  //Clients that need a different default (e.g. TritonClient) can override it here,
  //since only one place may declare "Retry" on a given ParameterSetDescription.
  static void fillBasePSetDescription(edm::ParameterSetDescription& desc,
                                      const std::string& defaultRetryType = "RetrySameServerAction");

protected:
  void setMode(SonicMode mode);
  void setUserMode(const std::string& userMode);

  virtual void evaluate() = 0;

  void start(edm::WaitingTaskWithArenaHolder holder);

  void start();

  void finish(bool success, std::exception_ptr eptr = std::exception_ptr{});

  //members
  SonicMode mode_;
  bool verbose_;
  std::unique_ptr<SonicDispatcher> dispatcher_;
  unsigned totalTries_;
  std::optional<edm::WaitingTaskWithArenaHolder> holder_;

  // Use a unique_ptr with a custom deleter to avoid incomplete type issues
  struct RetryDeleter {
    void operator()(RetryActionBase* ptr) const;
  };

  using RetryActionPtr = std::unique_ptr<RetryActionBase, RetryDeleter>;
  std::vector<RetryActionPtr> retryActions_;

  //for logging/debugging
  std::string debugName_, clientName_, fullDebugName_;
  //remember what user set at config time
  std::string userMode_;

  friend class SonicDispatcher;
  friend class SonicDispatcherPseudoAsync;
  friend class RetryActionBase;
};

#endif
