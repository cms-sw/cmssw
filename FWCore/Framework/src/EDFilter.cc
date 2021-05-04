/*----------------------------------------------------------------------
  

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/src/edmodule_mightGet_config.h"
#include "FWCore/Framework/src/EventSignalsSentry.h"
#include "FWCore/Framework/src/TransitionInfoTypes.h"
#include "FWCore/ServiceRegistry/interface/ESParentContext.h"

#include "SharedResourcesRegistry.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {

  EDFilter::EDFilter() : ProducerBase(), moduleDescription_(), previousParentage_(), previousParentageId_() {
    SharedResourcesRegistry::instance()->registerSharedResource(SharedResourcesRegistry::kLegacyModuleResourceName);
  }

  EDFilter::~EDFilter() {}

  bool EDFilter::doEvent(EventTransitionInfo const& info, ActivityRegistry* act, ModuleCallingContext const* mcc) {
    bool rc = false;
    Event e(info, moduleDescription_, mcc);
    e.setConsumer(this);
    e.setProducer(this, &previousParentage_);
    e.setSharedResourcesAcquirer(&resourceAcquirer_);
    EventSignalsSentry sentry(act, mcc);
    ESParentContext parentC(mcc);
    rc = this->filter(
        e,
        EventSetup{
            info, static_cast<unsigned int>(Transition::Event), esGetTokenIndices(Transition::Event), parentC, false});
    commit_(e, &previousParentageId_);
    return rc;
  }

  void EDFilter::doBeginJob() {
    std::vector<std::string> res = {SharedResourcesRegistry::kLegacyModuleResourceName};
    resourceAcquirer_ = SharedResourcesRegistry::instance()->createAcquirer(res);

    this->beginJob();
  }

  void EDFilter::doEndJob() { this->endJob(); }

  void EDFilter::doBeginRun(RunTransitionInfo const& info, ModuleCallingContext const* mcc) {
    Run r(info, moduleDescription_, mcc, false);
    r.setConsumer(this);
    Run const& cnstR = r;
    ESParentContext parentC(mcc);
    this->beginRun(cnstR,
                   EventSetup{info,
                              static_cast<unsigned int>(Transition::BeginRun),
                              esGetTokenIndices(Transition::BeginRun),
                              parentC,
                              false});
    commit_(r);
    return;
  }

  void EDFilter::doEndRun(RunTransitionInfo const& info, ModuleCallingContext const* mcc) {
    Run r(info, moduleDescription_, mcc, true);
    r.setConsumer(this);
    Run const& cnstR = r;
    ESParentContext parentC(mcc);
    this->endRun(
        cnstR,
        EventSetup{
            info, static_cast<unsigned int>(Transition::EndRun), esGetTokenIndices(Transition::EndRun), parentC, false});
    commit_(r);
    return;
  }

  void EDFilter::doBeginLuminosityBlock(LumiTransitionInfo const& info, ModuleCallingContext const* mcc) {
    LuminosityBlock lb(info, moduleDescription_, mcc, false);
    lb.setConsumer(this);
    LuminosityBlock const& cnstLb = lb;
    ESParentContext parentC(mcc);
    this->beginLuminosityBlock(cnstLb,
                               EventSetup{info,
                                          static_cast<unsigned int>(Transition::BeginLuminosityBlock),
                                          esGetTokenIndices(Transition::BeginLuminosityBlock),
                                          parentC,
                                          false});
    commit_(lb);
  }

  void EDFilter::doEndLuminosityBlock(LumiTransitionInfo const& info, ModuleCallingContext const* mcc) {
    LuminosityBlock lb(info, moduleDescription_, mcc, true);
    lb.setConsumer(this);
    LuminosityBlock const& cnstLb = lb;
    ESParentContext parentC(mcc);
    this->endLuminosityBlock(cnstLb,
                             EventSetup{info,
                                        static_cast<unsigned int>(Transition::EndLuminosityBlock),
                                        esGetTokenIndices(Transition::EndLuminosityBlock),
                                        parentC,
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
