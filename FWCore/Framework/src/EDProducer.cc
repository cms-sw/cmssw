/*----------------------------------------------------------------------
  

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/src/edmodule_mightGet_config.h"
#include "FWCore/Framework/src/EventSignalsSentry.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "SharedResourcesRegistry.h"

namespace edm {
  EDProducer::EDProducer() : ProducerBase(), moduleDescription_(), previousParentage_(), previousParentageId_() {
    SharedResourcesRegistry::instance()->registerSharedResource(SharedResourcesRegistry::kLegacyModuleResourceName);
  }

  EDProducer::~EDProducer() {}

  bool EDProducer::doEvent(EventPrincipal const& ep,
                           EventSetupImpl const& ci,
                           ActivityRegistry* act,
                           ModuleCallingContext const* mcc) {
    Event e(ep, moduleDescription_, mcc);
    e.setConsumer(this);
    e.setProducer(this, &previousParentage_);
    e.setSharedResourcesAcquirer(&resourceAcquirer_);
    EventSignalsSentry sentry(act, mcc);
    const EventSetup c{ci, static_cast<unsigned int>(Transition::Event), esGetTokenIndices(Transition::Event), false};
    this->produce(e, c);
    commit_(e, &previousParentageId_);
    return true;
  }

  void EDProducer::doBeginJob() {
    std::vector<std::string> res = {SharedResourcesRegistry::kLegacyModuleResourceName};
    resourceAcquirer_ = SharedResourcesRegistry::instance()->createAcquirer(res);
    this->beginJob();
  }

  void EDProducer::doEndJob() { this->endJob(); }

  void EDProducer::doBeginRun(RunPrincipal const& rp, EventSetupImpl const& ci, ModuleCallingContext const* mcc) {
    Run r(rp, moduleDescription_, mcc, false);
    r.setConsumer(this);
    Run const& cnstR = r;
    const EventSetup c{
        ci, static_cast<unsigned int>(Transition::BeginRun), esGetTokenIndices(Transition::BeginRun), false};
    this->beginRun(cnstR, c);
    commit_(r);
  }

  void EDProducer::doEndRun(RunPrincipal const& rp, EventSetupImpl const& ci, ModuleCallingContext const* mcc) {
    Run r(rp, moduleDescription_, mcc, true);
    r.setConsumer(this);
    Run const& cnstR = r;
    const EventSetup c{ci, static_cast<unsigned int>(Transition::EndRun), esGetTokenIndices(Transition::EndRun), false};
    this->endRun(cnstR, c);
    commit_(r);
  }

  void EDProducer::doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                          EventSetupImpl const& ci,
                                          ModuleCallingContext const* mcc) {
    LuminosityBlock lb(lbp, moduleDescription_, mcc, false);
    lb.setConsumer(this);
    LuminosityBlock const& cnstLb = lb;
    const EventSetup c{ci,
                       static_cast<unsigned int>(Transition::BeginLuminosityBlock),
                       esGetTokenIndices(Transition::BeginLuminosityBlock),
                       false};
    this->beginLuminosityBlock(cnstLb, c);
    commit_(lb);
  }

  void EDProducer::doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                        EventSetupImpl const& ci,
                                        ModuleCallingContext const* mcc) {
    LuminosityBlock lb(lbp, moduleDescription_, mcc, true);
    lb.setConsumer(this);
    const EventSetup c{ci,
                       static_cast<unsigned int>(Transition::EndLuminosityBlock),
                       esGetTokenIndices(Transition::EndLuminosityBlock),
                       false};
    LuminosityBlock const& cnstLb = lb;
    this->endLuminosityBlock(cnstLb, c);
    commit_(lb);
  }

  void EDProducer::doRespondToOpenInputFile(FileBlock const& fb) { respondToOpenInputFile(fb); }

  void EDProducer::doRespondToCloseInputFile(FileBlock const& fb) { respondToCloseInputFile(fb); }

  void EDProducer::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.setUnknown();
    descriptions.addDefault(desc);
  }

  void EDProducer::prevalidate(ConfigurationDescriptions& iConfig) { edmodule_mightGet_config(iConfig); }

  static const std::string kBaseType("EDProducer");

  const std::string& EDProducer::baseType() { return kBaseType; }
}  // namespace edm
