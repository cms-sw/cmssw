#ifndef FWCore_Framework_TransitionInfoTypes_h
#define FWCore_Framework_TransitionInfoTypes_h
//
// Package:     FWCore/Framework
//
/**

 Description: The types here are used by
 the functions beginGlobalTransitionAsync and
 endGlobalTransitionAsync. They hold some of the data
 passed as input arguments to those functions.

*/
//
// Original Author:  W. David Dagenhart
//         Created:  26 June 2020

#include <memory>
#include <vector>

namespace edm {
  class EventSetupImpl;
  class LuminosityBlockPrincipal;
  class ProcessBlockPrincipal;
  class RunPrincipal;

  class LumiTransitionInfo {
  public:
    LumiTransitionInfo(LuminosityBlockPrincipal& iPrincipal,
                       EventSetupImpl const& iEventSetupImpl,
                       std::vector<std::shared_ptr<const EventSetupImpl>> const* iEventSetupImpls)
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
    EventSetupImpl const& eventSetupImpl() const;

  private:
    ProcessBlockPrincipal& processBlockPrincipal_;
  };

};  // namespace edm

#endif
