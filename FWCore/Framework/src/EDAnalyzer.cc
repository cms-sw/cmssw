/*----------------------------------------------------------------------
  

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/src/edmodule_mightGet_config.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/Framework/src/EventSignalsSentry.h"

#include "SharedResourcesRegistry.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Provenance/interface/ProductRegistry.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

namespace edm {
  EDAnalyzer::~EDAnalyzer() {}
  EDAnalyzer::EDAnalyzer() : moduleDescription_() {
    SharedResourcesRegistry::instance()->registerSharedResource(SharedResourcesRegistry::kLegacyModuleResourceName);
  }

  bool EDAnalyzer::doEvent(EventPrincipal const& ep,
                           EventSetupImpl const& ci,
                           ActivityRegistry* act,
                           ModuleCallingContext const* mcc) {
    Event e(ep, moduleDescription_, mcc);
    e.setConsumer(this);
    e.setSharedResourcesAcquirer(&resourceAcquirer_);
    EventSignalsSentry sentry(act, mcc);
    const EventSetup c{ci, static_cast<unsigned int>(Transition::Event), esGetTokenIndices(Transition::Event), false};
    this->analyze(e, c);
    return true;
  }

  void EDAnalyzer::doBeginJob() {
    std::vector<std::string> res = {SharedResourcesRegistry::kLegacyModuleResourceName};
    resourceAcquirer_ = SharedResourcesRegistry::instance()->createAcquirer(res);

    this->beginJob();
  }

  void EDAnalyzer::doEndJob() { this->endJob(); }

  bool EDAnalyzer::doBeginRun(RunPrincipal const& rp, EventSetupImpl const& ci, ModuleCallingContext const* mcc) {
    Run r(rp, moduleDescription_, mcc, false);
    r.setConsumer(this);
    const EventSetup c{
        ci, static_cast<unsigned int>(Transition::BeginRun), esGetTokenIndices(Transition::BeginRun), false};
    this->beginRun(r, c);
    return true;
  }

  bool EDAnalyzer::doEndRun(RunPrincipal const& rp, EventSetupImpl const& ci, ModuleCallingContext const* mcc) {
    Run r(rp, moduleDescription_, mcc, true);
    r.setConsumer(this);
    const EventSetup c{ci, static_cast<unsigned int>(Transition::EndRun), esGetTokenIndices(Transition::EndRun), false};
    this->endRun(r, c);
    return true;
  }

  bool EDAnalyzer::doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                          EventSetupImpl const& ci,
                                          ModuleCallingContext const* mcc) {
    LuminosityBlock lb(lbp, moduleDescription_, mcc, false);
    lb.setConsumer(this);
    const EventSetup c{ci,
                       static_cast<unsigned int>(Transition::BeginLuminosityBlock),
                       esGetTokenIndices(Transition::BeginLuminosityBlock),
                       false};
    this->beginLuminosityBlock(lb, c);
    return true;
  }

  bool EDAnalyzer::doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                        EventSetupImpl const& ci,
                                        ModuleCallingContext const* mcc) {
    LuminosityBlock lb(lbp, moduleDescription_, mcc, true);
    lb.setConsumer(this);
    const EventSetup c{ci,
                       static_cast<unsigned int>(Transition::EndLuminosityBlock),
                       esGetTokenIndices(Transition::EndLuminosityBlock),
                       false};
    this->endLuminosityBlock(lb, c);
    return true;
  }

  void EDAnalyzer::doRespondToOpenInputFile(FileBlock const& fb) { respondToOpenInputFile(fb); }

  void EDAnalyzer::doRespondToCloseInputFile(FileBlock const& fb) { respondToCloseInputFile(fb); }

  void EDAnalyzer::callWhenNewProductsRegistered(std::function<void(BranchDescription const&)> const& func) {
    callWhenNewProductsRegistered_ = func;
  }

  void EDAnalyzer::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.setUnknown();
    descriptions.addDefault(desc);
  }

  void EDAnalyzer::prevalidate(ConfigurationDescriptions& iConfig) { edmodule_mightGet_config(iConfig); }

  void EDAnalyzer::registerProductsAndCallbacks(EDAnalyzer const*, ProductRegistry* reg) {
    if (callWhenNewProductsRegistered_) {
      reg->callForEachBranch(callWhenNewProductsRegistered_);

      Service<ConstProductRegistry> regService;
      regService->watchProductAdditions(callWhenNewProductsRegistered_);
    }
  }

  static const std::string kBaseType("EDAnalyzer");
  const std::string& EDAnalyzer::baseType() { return kBaseType; }
}  // namespace edm
