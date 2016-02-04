/*----------------------------------------------------------------------
  

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/src/CPCSentry.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {
  EDProducer::EDProducer() :
      ProducerBase(),
      moduleDescription_(),
      current_context_(0),
      previousParentage_(),
      previousParentageId_() { }

  EDProducer::~EDProducer() { }

  bool
  EDProducer::doEvent(EventPrincipal& ep, EventSetup const& c,
			     CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    Event e(ep, moduleDescription_);
    this->produce(e, c);
    e.commit_(&previousParentage_, &previousParentageId_);
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

  bool
  EDProducer::doBeginRun(RunPrincipal& rp, EventSetup const& c,
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    Run r(rp, moduleDescription_);
    this->beginRun(r, c);
    r.commit_();
    return true;
  }

  bool
  EDProducer::doEndRun(RunPrincipal& rp, EventSetup const& c,
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    Run r(rp, moduleDescription_);
    this->endRun(r, c);
    r.commit_();
    return true;
  }

  bool
  EDProducer::doBeginLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    LuminosityBlock lb(lbp, moduleDescription_);
    this->beginLuminosityBlock(lb, c);
    lb.commit_();
    return true;
  }

  bool
  EDProducer::doEndLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    LuminosityBlock lb(lbp, moduleDescription_);
    this->endLuminosityBlock(lb, c);
    lb.commit_();
    return true;
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
  EDProducer::doRespondToOpenOutputFiles(FileBlock const& fb) {
    respondToOpenOutputFiles(fb);
  }

  void
  EDProducer::doRespondToCloseOutputFiles(FileBlock const& fb) {
    respondToCloseOutputFiles(fb);
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
  
  static const std::string kBaseType("EDProducer");
  
  const std::string&
  EDProducer::baseType() {
    return kBaseType;
  }
}
