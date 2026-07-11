#ifndef FWCore_Framework_ComponentInterfaceHolder_h
#define FWCore_Framework_ComponentInterfaceHolder_h
// -*- C++ -*-
// Package:     Framework
// Class  :     ComponentInterfaceHolder
//
/**\class edm::eventsetup::ComponentInterfaceHolder 

 Description: Holds the interface of an EventSetup component along with the signals to be sent before and after construction of the component.

 Usage:
  Used by the ComponentFactory and EventSetup system.
*/
//
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProductResolverProvider.h"
#include "FWCore/Utilities/interface/Signal.h"

#include <memory>

namespace edm::eventsetup {
  struct ComponentDescription;

  class ComponentInterfaceHolder {
  public:
    void setFinder(std::shared_ptr<EventSetupRecordIntervalFinder> iFinder) { finder_ = std::move(iFinder); }
    void setProvider(std::shared_ptr<ESProductResolverProvider> iProvider) { provider_ = std::move(iProvider); }

    std::shared_ptr<EventSetupRecordIntervalFinder> finder() const { return finder_; }
    std::shared_ptr<ESProductResolverProvider> provider() const { return provider_; }

    signalslot::Signal<void(ComponentDescription const&)> const& preConstructionSignal() const {
      return preConstructionSignal_;
    }
    signalslot::Signal<void(ComponentDescription const&)> const& postConstructionSignal() const {
      return postConstructionSignal_;
    }

    void connectSignals(signalslot::Signal<void(ComponentDescription const&)> const& iPreConstructionSignal,
                        signalslot::Signal<void(ComponentDescription const&)> const& iPostConstructionSignal) {
      preConstructionSignal_.connect(std::cref(iPreConstructionSignal));
      postConstructionSignal_.connect(std::cref(iPostConstructionSignal));
    }

  private:
    std::shared_ptr<EventSetupRecordIntervalFinder> finder_;
    std::shared_ptr<ESProductResolverProvider> provider_;

    signalslot::Signal<void(ComponentDescription const&)> preConstructionSignal_;
    signalslot::Signal<void(ComponentDescription const&)> postConstructionSignal_;
  };
}  // namespace edm::eventsetup
#endif
