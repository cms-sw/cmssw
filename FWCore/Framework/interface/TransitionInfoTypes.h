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

namespace edm {
  class EventPrincipal;
  class EventSetupImpl;
  class LuminosityBlockPrincipal;
  class ProcessBlockPrincipal;
  class RunPrincipal;

  class EventTransitionInfo {
  public:
    EventTransitionInfo() {}

    EventTransitionInfo(EventPrincipal& iPrincipal, EventSetupImpl const& iEventSetupImpl)
        : eventPrincipal_(&iPrincipal), eventSetupImpl_(&iEventSetupImpl) {}

    EventPrincipal& principal() { return *eventPrincipal_; }
    EventPrincipal const& principal() const { return *eventPrincipal_; }
    EventSetupImpl const& eventSetupImpl() const { return *eventSetupImpl_; }

  private:
    EventPrincipal* eventPrincipal_ = nullptr;
    EventSetupImpl const* eventSetupImpl_ = nullptr;
  };

  class LumiTransitionInfo {
  public:
    LumiTransitionInfo() {}

    LumiTransitionInfo(LuminosityBlockPrincipal& iPrincipal, EventSetupImpl const& iEventSetupImpl)
        : luminosityBlockPrincipal_(&iPrincipal), eventSetupImpl_(&iEventSetupImpl) {}

    LuminosityBlockPrincipal& principal() { return *luminosityBlockPrincipal_; }
    LuminosityBlockPrincipal const& principal() const { return *luminosityBlockPrincipal_; }
    EventSetupImpl const& eventSetupImpl() const { return *eventSetupImpl_; }

  private:
    LuminosityBlockPrincipal* luminosityBlockPrincipal_ = nullptr;
    EventSetupImpl const* eventSetupImpl_ = nullptr;
  };

  class RunTransitionInfo {
  public:
    RunTransitionInfo() {}

    RunTransitionInfo(RunPrincipal& iPrincipal, EventSetupImpl const& iEventSetupImpl)
        : runPrincipal_(&iPrincipal), eventSetupImpl_(&iEventSetupImpl) {}

    RunPrincipal& principal() { return *runPrincipal_; }
    RunPrincipal const& principal() const { return *runPrincipal_; }
    EventSetupImpl const& eventSetupImpl() const { return *eventSetupImpl_; }

  private:
    RunPrincipal* runPrincipal_ = nullptr;
    EventSetupImpl const* eventSetupImpl_ = nullptr;
  };

  class ProcessBlockTransitionInfo {
  public:
    ProcessBlockTransitionInfo() {}

    ProcessBlockTransitionInfo(ProcessBlockPrincipal& iPrincipal) : processBlockPrincipal_(&iPrincipal) {}

    ProcessBlockPrincipal& principal() { return *processBlockPrincipal_; }
    ProcessBlockPrincipal const& principal() const { return *processBlockPrincipal_; }

  private:
    ProcessBlockPrincipal* processBlockPrincipal_ = nullptr;
  };

};  // namespace edm

#endif
