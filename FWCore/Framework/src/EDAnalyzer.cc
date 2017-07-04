/*----------------------------------------------------------------------
  

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/src/edmodule_mightGet_config.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/Framework/src/EventSignalsSentry.h"

#include "SharedResourcesRegistry.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Provenance/interface/ProductRegistry.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

namespace edm {
  EDAnalyzer::~EDAnalyzer() {
  }
  EDAnalyzer::EDAnalyzer() : moduleDescription_() {
    SharedResourcesRegistry::instance()->registerSharedResource(
                                                                SharedResourcesRegistry::kLegacyModuleResourceName);
  }


  bool
  EDAnalyzer::doEvent(EventPrincipal const& ep, EventSetup const& c,
                      ActivityRegistry* act,
                      ModuleCallingContext const* mcc) {
    Event e(ep, moduleDescription_, mcc);
    e.setConsumer(this);
    e.setSharedResourcesAcquirer(&resourceAcquirer_);
    EventSignalsSentry sentry(act,mcc);
    this->analyze(e, c);
    return true;
  }

  void
  EDAnalyzer::doBeginJob() {
    std::vector<std::string> res = {SharedResourcesRegistry::kLegacyModuleResourceName};
    resourceAcquirer_ = SharedResourcesRegistry::instance()->createAcquirer(res);

    this->beginJob();
  }

  void 
  EDAnalyzer::doEndJob() {
    this->endJob();
  }

  bool
  EDAnalyzer::doBeginRun(RunPrincipal const& rp, EventSetup const& c,
                         ModuleCallingContext const* mcc) {
    Run r(rp, moduleDescription_, mcc);
    r.setConsumer(this);
    this->beginRun(r, c);
    return true;
  }

  bool
  EDAnalyzer::doEndRun(RunPrincipal const& rp, EventSetup const& c,
                       ModuleCallingContext const* mcc) {
    Run r(rp, moduleDescription_, mcc);
    r.setConsumer(this);
    this->endRun(r, c);
    return true;
  }

  bool
  EDAnalyzer::doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                                     ModuleCallingContext const* mcc) {
    LuminosityBlock lb(lbp, moduleDescription_, mcc);
    lb.setConsumer(this);
    this->beginLuminosityBlock(lb, c);
    return true;
  }

  bool
  EDAnalyzer::doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                                   ModuleCallingContext const* mcc) {
    LuminosityBlock lb(lbp, moduleDescription_, mcc);
    lb.setConsumer(this);
    this->endLuminosityBlock(lb, c);
    return true;
  }

  void
  EDAnalyzer::doRespondToOpenInputFile(FileBlock const& fb) {
    respondToOpenInputFile(fb);
  }

  void
  EDAnalyzer::doRespondToCloseInputFile(FileBlock const& fb) {
    respondToCloseInputFile(fb);
  }

  void
  EDAnalyzer::callWhenNewProductsRegistered(std::function<void(BranchDescription const&)> const& func) {
    callWhenNewProductsRegistered_ = func;
  }

  void
  EDAnalyzer::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.setUnknown();
    descriptions.addDefault(desc);
  }
  
  void
  EDAnalyzer::prevalidate(ConfigurationDescriptions& iConfig) {
    edmodule_mightGet_config(iConfig);
  }

  void
  EDAnalyzer::registerProductsAndCallbacks(EDAnalyzer const*, ProductRegistry* reg) {

    if (callWhenNewProductsRegistered_) {

       reg->callForEachBranch(callWhenNewProductsRegistered_);

       Service<ConstProductRegistry> regService;
       regService->watchProductAdditions(callWhenNewProductsRegistered_);
    }
  }

  static const std::string kBaseType("EDAnalyzer");
  const std::string&
  EDAnalyzer::baseType() {
    return kBaseType;
  }
}
