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

    void beginRun(RunPrincipal& run, const edm::EventSetup& setup, ModuleCallingContext const*);
    void beginLuminosityBlock(LuminosityBlockPrincipal& lumi, const edm::EventSetup& setup, ModuleCallingContext const*);

    void endRun(RunPrincipal& run, const edm::EventSetup& setup, ModuleCallingContext const*);
    void endLuminosityBlock(LuminosityBlockPrincipal& lumi, const edm::EventSetup& setup, ModuleCallingContext const*);

    void setupPileUpEvent(EventPrincipal& ep, const EventSetup& setup);

    void beginStream(edm::StreamID iID, StreamContext& sContext) {workerManager_.beginStream(iID, sContext);}
    void endStream(edm::StreamID iID, StreamContext& sContext) {workerManager_.endStream(iID, sContext);}

  private:
    std::unique_ptr<ExceptionToActionTable> exceptionToActionTable_;
    WorkerManager workerManager_;
  };
}
#endif
