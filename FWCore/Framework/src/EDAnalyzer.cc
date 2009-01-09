/*----------------------------------------------------------------------
  

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/src/CPCSentry.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm
{
  EDAnalyzer::~EDAnalyzer()
  { }

  bool
  EDAnalyzer::doEvent(EventPrincipal const& ep, EventSetup const& c,
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    Event e(const_cast<EventPrincipal &>(ep), moduleDescription_);
    this->analyze(e, c);
    return true;
  }

  void
  EDAnalyzer::doBeginJob(EventSetup const& es) {
    this->beginJob(es);
  }
  
  void 
  EDAnalyzer::doEndJob() {
    this->endJob();
  }

  bool
  EDAnalyzer::doBeginRun(RunPrincipal const& rp, EventSetup const& c,
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    Run r(const_cast<RunPrincipal &>(rp), moduleDescription_);
    this->beginRun(r, c);
    return true;
  }

  bool
  EDAnalyzer::doEndRun(RunPrincipal const& rp, EventSetup const& c,
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    Run r(const_cast<RunPrincipal &>(rp), moduleDescription_);
    this->endRun(r, c);
    return true;
  }

  bool
  EDAnalyzer::doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    LuminosityBlock lb(const_cast<LuminosityBlockPrincipal &>(lbp), moduleDescription_);
    this->beginLuminosityBlock(lb, c);
    return true;
  }

  bool
  EDAnalyzer::doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
			CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    LuminosityBlock lb(const_cast<LuminosityBlockPrincipal &>(lbp), moduleDescription_);
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
  EDAnalyzer::doRespondToOpenOutputFiles(FileBlock const& fb) {
    respondToOpenOutputFiles(fb);
  }

  void
  EDAnalyzer::doRespondToCloseOutputFiles(FileBlock const& fb) {
    respondToCloseOutputFiles(fb);
  }

  CurrentProcessingContext const*
  EDAnalyzer::currentContext() const {
    return current_context_;
  }

  void
  EDAnalyzer::fillDescriptions(ConfigurationDescriptions & descriptions) {
    ParameterSetDescription desc;
    desc.setUnknown();
    descriptions.addUnknownLabel(desc);
  }

  std::string
  EDAnalyzer::baseType() {
    return std::string("EDAnalyzer");
  }
}
