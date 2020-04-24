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

    void beginRun(RunPrincipal& run, const edm::EventSetup& setup, ModuleCallingContext const*, StreamContext& sContext);
    void beginLuminosityBlock(LuminosityBlockPrincipal& lumi, const edm::EventSetup& setup, ModuleCallingContext const*, StreamContext& sContext);

    void endRun(RunPrincipal& run, const edm::EventSetup& setup, ModuleCallingContext const*, StreamContext& sContext);
    void endLuminosityBlock(LuminosityBlockPrincipal& lumi, const edm::EventSetup& setup, ModuleCallingContext const*, StreamContext& sContext);

    void setupPileUpEvent(EventPrincipal& ep, const EventSetup& setup, StreamContext& sContext);

    void beginJob(ProductRegistry const& iRegistry) {workerManager_.beginJob(iRegistry);}
    void endJob() {workerManager_.endJob();}

    void beginStream(edm::StreamID iID, StreamContext& sContext);
    void endStream(edm::StreamID iID, StreamContext& sContext);

  private:
    std::unique_ptr<ExceptionToActionTable> exceptionToActionTable_;
    WorkerManager workerManager_;
  };
}
#endif
