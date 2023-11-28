#ifndef Mixing_Base_SecondaryEventProvider_h
#define Mixing_Base_SecondaryEventProvider_h

#include "FWCore/Framework/interface/WorkerManager.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include <memory>
#include <string>
#include <vector>

namespace edm {
  class ModuleCallingContext;

  class SecondaryEventProvider {
  public:
    SecondaryEventProvider(std::vector<ParameterSet>& psets,
                           ProductRegistry& pregistry,
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

    void beginJob(ProductRegistry const& iRegistry, eventsetup::ESRecordsToProductResolverIndices const&);
    void endJob() { workerManager_.endJob(); }

    void beginStream(edm::StreamID iID, StreamContext& sContext);
    void endStream(edm::StreamID iID, StreamContext& sContext);

  private:
    std::unique_ptr<ExceptionToActionTable> exceptionToActionTable_;
    WorkerManager workerManager_;
  };
}  // namespace edm
#endif
