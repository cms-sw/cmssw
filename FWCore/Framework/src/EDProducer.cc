/*----------------------------------------------------------------------
  

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/src/CPCSentry.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/src/edmodule_mightGet_config.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {
  EDProducer::EDProducer() :
      ProducerBase(),
      moduleDescription_(),
      current_context_(nullptr),
      previousParentage_(),
      previousParentageId_() { }

  EDProducer::~EDProducer() { }

  bool
  EDProducer::doEvent(EventPrincipal& ep, EventSetup const& c,
			     CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    Event e(ep, moduleDescription_);
    e.setConsumer(this);
    this->produce(e, c);
    commit_(e, &previousParentage_, &previousParentageId_);
    return true;
  }

  void 
  EDProducer::doBeginJob() {
    this->beginJob();
  }
  
  void 
  EDProducer::doEndJob() {
    this->endJob();
  }

  void
  EDProducer::doBeginRun(RunPrincipal& rp, EventSetup const& c,
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    Run r(rp, moduleDescription_);
    r.setConsumer(this);
    Run const& cnstR = r;
    this->beginRun(cnstR, c);
    commit_(r);
  }

  void
  EDProducer::doEndRun(RunPrincipal& rp, EventSetup const& c,
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    Run r(rp, moduleDescription_);
    r.setConsumer(this);
    Run const& cnstR = r;
    this->endRun(cnstR, c);
    commit_(r);
  }

  void
  EDProducer::doBeginLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    LuminosityBlock lb(lbp, moduleDescription_);
    lb.setConsumer(this);
    LuminosityBlock const& cnstLb = lb;
    this->beginLuminosityBlock(cnstLb, c);
    commit_(lb);
  }

  void
  EDProducer::doEndLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    LuminosityBlock lb(lbp, moduleDescription_);
    lb.setConsumer(this);
    LuminosityBlock const& cnstLb = lb;
    this->endLuminosityBlock(cnstLb, c);
    commit_(lb);
  }

  void
  EDProducer::doRespondToOpenInputFile(FileBlock const& fb) {
    respondToOpenInputFile(fb);
  }

  void
  EDProducer::doRespondToCloseInputFile(FileBlock const& fb) {
    respondToCloseInputFile(fb);
  }

  void 
  EDProducer::doPreForkReleaseResources() {
    preForkReleaseResources();
  }
  
  void 
  EDProducer::doPostForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren) {
    postForkReacquireResources(iChildIndex, iNumberOfChildren);
  }
  
  CurrentProcessingContext const*
  EDProducer::currentContext() const {
    return current_context_;
  }

  void
  EDProducer::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.setUnknown();
    descriptions.addDefault(desc);
  }
  
  void
  EDProducer::prevalidate(ConfigurationDescriptions& iConfig) {
    edmodule_mightGet_config(iConfig);
  }
  
  static const std::string kBaseType("EDProducer");
  
  const std::string&
  EDProducer::baseType() {
    return kBaseType;
  }
}
