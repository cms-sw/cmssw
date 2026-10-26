// After a given module has run, emit the delayed-get signals of that module from a new thread, which has never entered
// a TBB arena. This reproduces, in a simple job, a service being called on a thread that its TBB observer has not seen,
// as happens when the framework runs some work in an arena that the service does not observe.

#include <string>
#include <thread>

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"

namespace edmtest {

  class UnobservedThreadSignals {
  public:
    UnobservedThreadSignals(edm::ParameterSet const& config, edm::ActivityRegistry& registry)
        : moduleLabel_(config.getUntrackedParameter<std::string>("moduleLabel")), registry_(registry) {
      registry.watchPostModuleEvent(this, &UnobservedThreadSignals::postModuleEvent);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.addUntracked<std::string>("moduleLabel")
          ->setComment("Label of the module whose delayed-get signals are emitted from a new thread");
      descriptions.add("UnobservedThreadSignals", desc);
    }

  private:
    void postModuleEvent(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {
      if (mcc.moduleDescription()->moduleLabel() != moduleLabel_) {
        return;
      }
      // a delayed get made by the running module from a thread outside of the TBB arenas
      edm::ModuleCallingContext context(mcc);
      context.setState(edm::ModuleCallingContext::State::kRunning);
      std::thread thread([this, &sc, &context]() {
        registry_.preModuleEventDelayedGetSignal_.emit(sc, context);
        registry_.postModuleEventDelayedGetSignal_.emit(sc, context);
      });
      thread.join();
    }

    const std::string moduleLabel_;
    edm::ActivityRegistry& registry_;
  };

}  // namespace edmtest

using edmtest::UnobservedThreadSignals;
DEFINE_FWK_SERVICE(UnobservedThreadSignals);
