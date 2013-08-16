/*----------------------------------------------------------------------
  

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/src/CPCSentry.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/src/edmodule_mightGet_config.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {
  EDFilter::~EDFilter() {
  }

  bool
  EDFilter::doEvent(EventPrincipal& ep, EventSetup const& c,
		    CurrentProcessingContext const* cpc,
                    ModuleCallingContext const* mcc) {
    detail::CPCSentry sentry(current_context_, cpc);
    bool rc = false;
    Event e(ep, moduleDescription_, mcc);
    e.setConsumer(this);
    rc = this->filter(e, c);
    commit_(e,&previousParentage_, &previousParentageId_);
    return rc;
  }

  void 
  EDFilter::doBeginJob() { 
    this->beginJob();
  }
   
  void EDFilter::doEndJob() { 
    this->endJob();
  }

  void
  EDFilter::doBeginRun(RunPrincipal& rp, EventSetup const& c,
		       CurrentProcessingContext const* cpc,
                       ModuleCallingContext const* mcc) {
    detail::CPCSentry sentry(current_context_, cpc);
    Run r(rp, moduleDescription_, mcc);
    r.setConsumer(this);
    Run const& cnstR=r;
    this->beginRun(cnstR, c);
    commit_(r);
    return;
  }

  void
  EDFilter::doEndRun(RunPrincipal& rp, EventSetup const& c,
		     CurrentProcessingContext const* cpc,
                     ModuleCallingContext const* mcc) {
    detail::CPCSentry sentry(current_context_, cpc);
    Run r(rp, moduleDescription_, mcc);
    r.setConsumer(this);
    Run const& cnstR=r;
    this->endRun(cnstR, c);
    commit_(r);
    return;
  }

  void
  EDFilter::doBeginLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                                   CurrentProcessingContext const* cpc,
                                   ModuleCallingContext const* mcc) {
    detail::CPCSentry sentry(current_context_, cpc);
    LuminosityBlock lb(lbp, moduleDescription_, mcc);
    lb.setConsumer(this);
    LuminosityBlock const& cnstLb = lb;
    this->beginLuminosityBlock(cnstLb, c);
    commit_(lb);
  }

  void
  EDFilter::doEndLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
			         CurrentProcessingContext const* cpc,
                                 ModuleCallingContext const* mcc) {
    detail::CPCSentry sentry(current_context_, cpc);
    LuminosityBlock lb(lbp, moduleDescription_, mcc);
    lb.setConsumer(this);
    LuminosityBlock const& cnstLb = lb;
    this->endLuminosityBlock(cnstLb, c);
    commit_(lb);
    return ;
  }

  void
  EDFilter::doRespondToOpenInputFile(FileBlock const& fb) {
    respondToOpenInputFile(fb);
  }

  void
  EDFilter::doRespondToCloseInputFile(FileBlock const& fb) {
    respondToCloseInputFile(fb);
  }

  void 
  EDFilter::doPreForkReleaseResources() {
    preForkReleaseResources();
  }
  
  void 
  EDFilter::doPostForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren) {
    postForkReacquireResources(iChildIndex, iNumberOfChildren);
  }
  
  CurrentProcessingContext const*
  EDFilter::currentContext() const {
    return current_context_;
  }
  
  void
  EDFilter::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.setUnknown();
    descriptions.addDefault(desc);
  }

  void
  EDFilter::prevalidate(ConfigurationDescriptions& iConfig) {
    edmodule_mightGet_config(iConfig);
  }
  

  static const std::string kBaseType("EDFilter");
  const std::string&
  EDFilter::baseType() {
    return kBaseType;
  }
}
