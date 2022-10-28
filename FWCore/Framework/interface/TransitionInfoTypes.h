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

    LumiTransitionInfo(LuminosityBlockPrincipal& iPrincipal,
                       EventSetupImpl const& iEventSetupImpl,
                       std::vector<std::shared_ptr<const EventSetupImpl>> const* iEventSetupImpls = nullptr)
        : luminosityBlockPrincipal_(&iPrincipal),
          eventSetupImpl_(&iEventSetupImpl),
          eventSetupImpls_(iEventSetupImpls) {}

    LuminosityBlockPrincipal& principal() { return *luminosityBlockPrincipal_; }
    LuminosityBlockPrincipal const& principal() const { return *luminosityBlockPrincipal_; }
    EventSetupImpl const& eventSetupImpl() const { return *eventSetupImpl_; }
    std::vector<std::shared_ptr<const EventSetupImpl>> const* eventSetupImpls() const { return eventSetupImpls_; }

  private:
    LuminosityBlockPrincipal* luminosityBlockPrincipal_ = nullptr;
    EventSetupImpl const* eventSetupImpl_ = nullptr;
    // The first element of this vector refers to the top level process.
    // If there are SubProcesses, then each additional element refers to
    // one SubProcess. The previous data member refers to the same EventSetupImpl
    // object as one element of this vector (the one currently being handled).
    std::vector<std::shared_ptr<const EventSetupImpl>> const* eventSetupImpls_ = nullptr;
  };

  class RunTransitionInfo {
  public:
    RunTransitionInfo() {}

    RunTransitionInfo(RunPrincipal& iPrincipal,
                      EventSetupImpl const& iEventSetupImpl,
                      std::vector<std::shared_ptr<const EventSetupImpl>> const* iEventSetupImpls = nullptr)
        : runPrincipal_(&iPrincipal), eventSetupImpl_(&iEventSetupImpl), eventSetupImpls_(iEventSetupImpls) {}

    RunPrincipal& principal() { return *runPrincipal_; }
    RunPrincipal const& principal() const { return *runPrincipal_; }
    EventSetupImpl const& eventSetupImpl() const { return *eventSetupImpl_; }
    std::vector<std::shared_ptr<const EventSetupImpl>> const* eventSetupImpls() const { return eventSetupImpls_; }

  private:
    RunPrincipal* runPrincipal_ = nullptr;
    EventSetupImpl const* eventSetupImpl_ = nullptr;
    // The first element of this vector refers to the top level process.
    // If there are SubProcesses, then each additional element refers to
    // one SubProcess. The previous data member refers to the same EventSetupImpl
    // object as one element of this vector (the one currently being handled).
    std::vector<std::shared_ptr<const EventSetupImpl>> const* eventSetupImpls_ = nullptr;
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
