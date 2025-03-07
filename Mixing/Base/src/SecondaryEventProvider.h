#ifndef Mixing_Base_SecondaryEventProvider_h
#define Mixing_Base_SecondaryEventProvider_h

#include "FWCore/Framework/interface/WorkerManager.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistryfwd.h"

#include <memory>
#include <string>
#include <vector>

namespace edm {
  class ModuleCallingContext;

  class SecondaryEventProvider {
  public:
    SecondaryEventProvider(std::vector<ParameterSet>& psets,
                           SignallingProductRegistryFiller& pregistry,
                           std::shared_ptr<ProcessConfiguration> processConfiguration);

    void beginRun(RunPrincipal& run,
                  const edm::EventSetupImpl& setup,
                  ModuleCallingContext const*,
                  StreamContext& sContext);
    void beginLuminosityBlock(LuminosityBlockPrincipal& lumi,
                              const edm::EventSetupImpl& setup,
                              ModuleCallingContext const*,
                              StreamContext& sContext);

    void endRun(RunPrincipal& run,
                const edm::EventSetupImpl& setup,
                ModuleCallingContext const*,
                StreamContext& sContext);
    void endLuminosityBlock(LuminosityBlockPrincipal& lumi,
                            const edm::EventSetupImpl& setup,
                            ModuleCallingContext const*,
                            StreamContext& sContext);

    void setupPileUpEvent(EventPrincipal& ep, const EventSetupImpl& setup, StreamContext& sContext);

    void beginJob(ProductRegistry const& iRegistry,
                  eventsetup::ESRecordsToProductResolverIndices const&,
                  GlobalContext const&);
    void endJob(ExceptionCollector& exceptionCollector, GlobalContext const& globalContext) {
      workerManager_.endJob(exceptionCollector, globalContext);
    }

    void beginStream(edm::StreamID, StreamContext const&);
    void endStream(edm::StreamID, StreamContext const&, ExceptionCollector&);

  private:
    std::unique_ptr<ExceptionToActionTable> exceptionToActionTable_;
    WorkerManager workerManager_;
  };
}  // namespace edm
#endif
