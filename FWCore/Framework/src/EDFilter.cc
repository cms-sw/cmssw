/*----------------------------------------------------------------------
  

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/src/edmodule_mightGet_config.h"
#include "FWCore/Framework/src/EventSignalsSentry.h"

#include "SharedResourcesRegistry.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {

  EDFilter::EDFilter() : ProducerBase(), moduleDescription_(), previousParentage_(), previousParentageId_() {
    SharedResourcesRegistry::instance()->registerSharedResource(SharedResourcesRegistry::kLegacyModuleResourceName);
  }

  EDFilter::~EDFilter() {}

  bool EDFilter::doEvent(EventPrincipal const& ep,
                         EventSetupImpl const& c,
                         ActivityRegistry* act,
                         ModuleCallingContext const* mcc) {
    bool rc = false;
    Event e(ep, moduleDescription_, mcc);
    e.setConsumer(this);
    e.setProducer(this, &previousParentage_);
    e.setSharedResourcesAcquirer(&resourceAcquirer_);
    EventSignalsSentry sentry(act, mcc);
    rc = this->filter(
        e, EventSetup{c, static_cast<unsigned int>(Transition::Event), esGetTokenIndices(Transition::Event), false});
    commit_(e, &previousParentageId_);
    return rc;
  }

  void EDFilter::doBeginJob() {
    std::vector<std::string> res = {SharedResourcesRegistry::kLegacyModuleResourceName};
    resourceAcquirer_ = SharedResourcesRegistry::instance()->createAcquirer(res);

    this->beginJob();
  }

  void EDFilter::doEndJob() { this->endJob(); }

  void EDFilter::doBeginRun(RunPrincipal const& rp, EventSetupImpl const& c, ModuleCallingContext const* mcc) {
    Run r(rp, moduleDescription_, mcc, false);
    r.setConsumer(this);
    Run const& cnstR = r;
    this->beginRun(
        cnstR,
        EventSetup{c, static_cast<unsigned int>(Transition::BeginRun), esGetTokenIndices(Transition::BeginRun), false});
    commit_(r);
    return;
  }

  void EDFilter::doEndRun(RunPrincipal const& rp, EventSetupImpl const& c, ModuleCallingContext const* mcc) {
    Run r(rp, moduleDescription_, mcc, true);
    r.setConsumer(this);
    Run const& cnstR = r;
    this->endRun(
        cnstR,
        EventSetup{c, static_cast<unsigned int>(Transition::EndRun), esGetTokenIndices(Transition::EndRun), false});
    commit_(r);
    return;
  }

  void EDFilter::doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                        EventSetupImpl const& c,
                                        ModuleCallingContext const* mcc) {
    LuminosityBlock lb(lbp, moduleDescription_, mcc, false);
    lb.setConsumer(this);
    LuminosityBlock const& cnstLb = lb;
    this->beginLuminosityBlock(cnstLb,
                               EventSetup{c,
                                          static_cast<unsigned int>(Transition::BeginLuminosityBlock),
                                          esGetTokenIndices(Transition::BeginLuminosityBlock),
                                          false});
    commit_(lb);
  }

  void EDFilter::doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                      EventSetupImpl const& c,
                                      ModuleCallingContext const* mcc) {
    LuminosityBlock lb(lbp, moduleDescription_, mcc, true);
    lb.setConsumer(this);
    LuminosityBlock const& cnstLb = lb;
    this->endLuminosityBlock(cnstLb,
                             EventSetup{c,
                                        static_cast<unsigned int>(Transition::EndLuminosityBlock),
                                        esGetTokenIndices(Transition::EndLuminosityBlock),
                                        false});
    commit_(lb);
    return;
  }

  void EDFilter::doRespondToOpenInputFile(FileBlock const& fb) { respondToOpenInputFile(fb); }

  void EDFilter::doRespondToCloseInputFile(FileBlock const& fb) { respondToCloseInputFile(fb); }

  void EDFilter::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.setUnknown();
    descriptions.addDefault(desc);
  }

  void EDFilter::prevalidate(ConfigurationDescriptions& iConfig) { edmodule_mightGet_config(iConfig); }

  static const std::string kBaseType("EDFilter");
  const std::string& EDFilter::baseType() { return kBaseType; }
}  // namespace edm
