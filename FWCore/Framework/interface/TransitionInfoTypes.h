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
#include <typeindex>

namespace edm {
  class EventPrincipal;
  class EventSetupImpl;
  class LuminosityBlockPrincipal;
  class ProcessBlockPrincipal;
  class RunPrincipal;

  class EventTransitionInfo;
  class RunTransitionInfo;
  class LumiTransitionInfo;
  class ProcessBlockTransitionInfo;
  class InputProcessBlockTransitionInfo;
  class TransitionInfoKey {
  public:
    friend class EventTransitionInfo;
    friend class RunTransitionInfo;
    friend class LumiTransitionInfo;
    friend class ProcessBlockTransitionInfo;
    friend class InputProcessBlockTransitionInfo;

    std::type_index index() const noexcept { return index_; }

    TransitionInfoKey(const TransitionInfoKey&) = default;
    TransitionInfoKey& operator=(const TransitionInfoKey&) = default;
    TransitionInfoKey(TransitionInfoKey&&) = default;
    TransitionInfoKey& operator=(TransitionInfoKey&&) = default;

    bool operator==(const TransitionInfoKey& rhs) const noexcept { return index_ == rhs.index_; }
    std::strong_ordering operator<=>(const TransitionInfoKey& rhs) const noexcept { return index_ <=> rhs.index_; }

  private:
    TransitionInfoKey(std::type_index iIndex) : index_(iIndex) {}
    std::type_index index_;
  };
  class EventTransitionInfo {
  public:
    EventTransitionInfo() {}

    EventTransitionInfo(EventPrincipal& iPrincipal, EventSetupImpl const& iEventSetupImpl)
        : eventPrincipal_(&iPrincipal), eventSetupImpl_(&iEventSetupImpl) {}

    EventPrincipal& principal() { return *eventPrincipal_; }
    EventPrincipal const& principal() const { return *eventPrincipal_; }
    EventSetupImpl const& eventSetupImpl() const { return *eventSetupImpl_; }

    static TransitionInfoKey key() noexcept { return TransitionInfoKey(typeid(EventTransitionInfo)); }

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

    static TransitionInfoKey key() noexcept { return TransitionInfoKey(typeid(LumiTransitionInfo)); }

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

    static TransitionInfoKey key() noexcept { return TransitionInfoKey(typeid(RunTransitionInfo)); }

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

    static TransitionInfoKey key() noexcept { return TransitionInfoKey(typeid(ProcessBlockTransitionInfo)); }

  private:
    ProcessBlockPrincipal* processBlockPrincipal_ = nullptr;
  };

  class InputProcessBlockTransitionInfo {
  public:
    InputProcessBlockTransitionInfo() {}

    InputProcessBlockTransitionInfo(ProcessBlockPrincipal& iPrincipal) : processBlockPrincipal_(&iPrincipal) {}

    ProcessBlockPrincipal& principal() { return *processBlockPrincipal_; }
    ProcessBlockPrincipal const& principal() const { return *processBlockPrincipal_; }

    static TransitionInfoKey key() noexcept { return TransitionInfoKey(typeid(InputProcessBlockTransitionInfo)); }

  private:
    ProcessBlockPrincipal* processBlockPrincipal_ = nullptr;
  };

};  // namespace edm

#endif
