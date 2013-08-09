/*----------------------------------------------------------------------
  

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/src/CPCSentry.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/src/edmodule_mightGet_config.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Provenance/interface/ProductRegistry.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"

namespace edm {
  EDAnalyzer::~EDAnalyzer() {
  }

  bool
  EDAnalyzer::doEvent(EventPrincipal const& ep, EventSetup const& c,
		      CurrentProcessingContext const* cpc,
                      ModuleCallingContext const* mcc) {
    detail::CPCSentry sentry(current_context_, cpc);
    Event e(const_cast<EventPrincipal&>(ep), moduleDescription_, mcc);
    e.setConsumer(this);
    this->analyze(e, c);
    return true;
  }

  void
  EDAnalyzer::doBeginJob() {
    this->beginJob();
  }

  void 
  EDAnalyzer::doEndJob() {
    this->endJob();
  }

  bool
  EDAnalyzer::doBeginRun(RunPrincipal const& rp, EventSetup const& c,
			 CurrentProcessingContext const* cpc,
                         ModuleCallingContext const* mcc) {
    detail::CPCSentry sentry(current_context_, cpc);
    Run r(const_cast<RunPrincipal&>(rp), moduleDescription_, mcc);
    r.setConsumer(this);
    this->beginRun(r, c);
    return true;
  }

  bool
  EDAnalyzer::doEndRun(RunPrincipal const& rp, EventSetup const& c,
		       CurrentProcessingContext const* cpc,
                       ModuleCallingContext const* mcc) {
    detail::CPCSentry sentry(current_context_, cpc);
    Run r(const_cast<RunPrincipal&>(rp), moduleDescription_, mcc);
    r.setConsumer(this);
    this->endRun(r, c);
    return true;
  }

  bool
  EDAnalyzer::doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
			             CurrentProcessingContext const* cpc,
                                     ModuleCallingContext const* mcc) {
    detail::CPCSentry sentry(current_context_, cpc);
    LuminosityBlock lb(const_cast<LuminosityBlockPrincipal&>(lbp), moduleDescription_, mcc);
    lb.setConsumer(this);
    this->beginLuminosityBlock(lb, c);
    return true;
  }

  bool
  EDAnalyzer::doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                                   CurrentProcessingContext const* cpc,
                                   ModuleCallingContext const* mcc) {
    detail::CPCSentry sentry(current_context_, cpc);
    LuminosityBlock lb(const_cast<LuminosityBlockPrincipal&>(lbp), moduleDescription_, mcc);
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
  EDAnalyzer::doPreForkReleaseResources() {
    preForkReleaseResources();
  }
  
  void 
  EDAnalyzer::doPostForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren) {
    postForkReacquireResources(iChildIndex, iNumberOfChildren);
  }
   
  CurrentProcessingContext const*
  EDAnalyzer::currentContext() const {
    return current_context_;
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
