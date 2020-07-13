#ifndef FWCore_Framework_TransitionInfoTypes_h
#define FWCore_Framework_TransitionInfoTypes_h
//
// Package:     FWCore/Framework
//
/**

 Description: The types here are used to pass information
 down to the Worker class from the EventProcessor.

*/
//
// Original Author:  W. David Dagenhart
//         Created:  26 June 2020

#include <memory>
#include <vector>

namespace edm {
  class EventPrincipal;
  class EventSetupImpl;
  class LuminosityBlockPrincipal;
  class ProcessBlockPrincipal;
  class RunPrincipal;

  class EventTransitionInfo {
  public:
    EventTransitionInfo(EventPrincipal& iPrincipal, EventSetupImpl const& iEventSetupImpl)
        : eventPrincipal_(iPrincipal), eventSetupImpl_(iEventSetupImpl) {}

    EventPrincipal& principal() { return eventPrincipal_; }
    EventPrincipal const& principal() const { return eventPrincipal_; }
    EventSetupImpl const& eventSetupImpl() const { return eventSetupImpl_; }

  private:
    EventPrincipal& eventPrincipal_;
    EventSetupImpl const& eventSetupImpl_;
  };

  class LumiTransitionInfo {
  public:
    LumiTransitionInfo(LuminosityBlockPrincipal& iPrincipal,
                       EventSetupImpl const& iEventSetupImpl,
                       std::vector<std::shared_ptr<const EventSetupImpl>> const* iEventSetupImpls = nullptr)
        : luminosityBlockPrincipal_(iPrincipal), eventSetupImpl_(iEventSetupImpl), eventSetupImpls_(iEventSetupImpls) {}

    LuminosityBlockPrincipal& principal() { return luminosityBlockPrincipal_; }
    LuminosityBlockPrincipal const& principal() const { return luminosityBlockPrincipal_; }
    EventSetupImpl const& eventSetupImpl() const { return eventSetupImpl_; }
    std::vector<std::shared_ptr<const EventSetupImpl>> const* eventSetupImpls() const { return eventSetupImpls_; }

  private:
    LuminosityBlockPrincipal& luminosityBlockPrincipal_;
    EventSetupImpl const& eventSetupImpl_;
    std::vector<std::shared_ptr<const EventSetupImpl>> const* eventSetupImpls_;
  };

  class RunTransitionInfo {
  public:
    RunTransitionInfo(RunPrincipal& iPrincipal, EventSetupImpl const& iEventSetupImpl)
        : runPrincipal_(iPrincipal), eventSetupImpl_(iEventSetupImpl) {}

    RunPrincipal& principal() { return runPrincipal_; }
    RunPrincipal const& principal() const { return runPrincipal_; }
    EventSetupImpl const& eventSetupImpl() const { return eventSetupImpl_; }

  private:
    RunPrincipal& runPrincipal_;
    EventSetupImpl const& eventSetupImpl_;
  };

  class ProcessBlockTransitionInfo {
  public:
    ProcessBlockTransitionInfo(ProcessBlockPrincipal& iPrincipal) : processBlockPrincipal_(iPrincipal) {}

    ProcessBlockPrincipal& principal() { return processBlockPrincipal_; }
    ProcessBlockPrincipal const& principal() const { return processBlockPrincipal_; }

  private:
    ProcessBlockPrincipal& processBlockPrincipal_;
  };

};  // namespace edm

#endif
