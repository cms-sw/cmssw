#ifndef Mixing_Base_SecondaryEventProvider_h
#define Mixing_Base_SecondaryEventProvider_h

#include "FWCore/Framework/interface/WorkerManager.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "boost/shared_ptr.hpp"

#include <memory>
#include <string>
#include <vector>

namespace edm {
  class ModuleCallingContext;

  class SecondaryEventProvider {
  public:
    SecondaryEventProvider(std::vector<ParameterSet>& psets,
             ProductRegistry& pregistry,
             boost::shared_ptr<ProcessConfiguration> processConfiguration);

    void beginRun(RunPrincipal& run, const edm::EventSetup& setup, ModuleCallingContext const*);
    void beginLuminosityBlock(LuminosityBlockPrincipal& lumi, const edm::EventSetup& setup, ModuleCallingContext const*);

    void endRun(RunPrincipal& run, const edm::EventSetup& setup, ModuleCallingContext const*);
    void endLuminosityBlock(LuminosityBlockPrincipal& lumi, const edm::EventSetup& setup, ModuleCallingContext const*);

    void setupPileUpEvent(EventPrincipal& ep, const EventSetup& setup);

    void beginJob(ProductRegistry const& iRegistry) {workerManager_.beginJob(iRegistry);}
    void endJob() {workerManager_.endJob();}

  private:
    std::unique_ptr<ExceptionToActionTable> exceptionToActionTable_;
    WorkerManager workerManager_;
  };
}
#endif
