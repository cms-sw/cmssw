// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     one::EDFilterBase
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Thu, 02 May 2013 21:56:04 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/one/EDFilterBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/src/edmodule_mightGet_config.h"
#include "FWCore/Framework/src/PreallocationConfiguration.h"
#include "FWCore/Framework/src/EventSignalsSentry.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

//
// constants, enums and typedefs
//
namespace edm {
  class ModuleCallingContext;

  namespace one {
    //
    // static data member definitions
    //

    //
    // constructors and destructor
    //
    EDFilterBase::EDFilterBase() : ProducerBase(), moduleDescription_(), previousParentage_(), previousParentageId_() {}

    EDFilterBase::~EDFilterBase() {}

    bool EDFilterBase::doEvent(EventPrincipal const& ep,
                               EventSetupImpl const& ci,
                               ActivityRegistry* act,
                               ModuleCallingContext const* mcc) {
      Event e(ep, moduleDescription_, mcc);
      e.setConsumer(this);
      e.setProducer(this, &previousParentage_);
      bool returnValue = true;
      e.setSharedResourcesAcquirer(&resourcesAcquirer_);
      EventSignalsSentry sentry(act, mcc);
      const EventSetup c{ci, static_cast<unsigned int>(Transition::Event), esGetTokenIndices(Transition::Event), false};
      returnValue = this->filter(e, c);
      commit_(e, &previousParentageId_);
      return returnValue;
    }

    SharedResourcesAcquirer EDFilterBase::createAcquirer() {
      return SharedResourcesAcquirer{
          std::vector<std::shared_ptr<SerialTaskQueue>>(1, std::make_shared<SerialTaskQueue>())};
    }

    SerialTaskQueue* EDFilterBase::globalRunsQueue() { return nullptr; }
    SerialTaskQueue* EDFilterBase::globalLuminosityBlocksQueue() { return nullptr; };

    void EDFilterBase::doBeginJob() {
      resourcesAcquirer_ = createAcquirer();

      this->beginJob();
    }

    void EDFilterBase::doEndJob() { this->endJob(); }

    void EDFilterBase::doPreallocate(PreallocationConfiguration const& iPrealloc) {
      auto const nThreads = iPrealloc.numberOfThreads();
      preallocThreads(nThreads);
      preallocLumis(iPrealloc.numberOfLuminosityBlocks());
    }
    void EDFilterBase::preallocLumis(unsigned int){};

    void EDFilterBase::doBeginRun(RunPrincipal const& rp, EventSetupImpl const& ci, ModuleCallingContext const* mcc) {
      Run r(rp, moduleDescription_, mcc, false);
      r.setConsumer(this);
      Run const& cnstR = r;
      const EventSetup c{
          ci, static_cast<unsigned int>(Transition::BeginRun), esGetTokenIndices(Transition::BeginRun), false};
      this->doBeginRun_(cnstR, c);
      r.setProducer(this);
      this->doBeginRunProduce_(r, c);
      commit_(r);
    }

    void EDFilterBase::doEndRun(RunPrincipal const& rp, EventSetupImpl const& ci, ModuleCallingContext const* mcc) {
      Run r(rp, moduleDescription_, mcc, true);
      r.setConsumer(this);
      Run const& cnstR = r;
      const EventSetup c{
          ci, static_cast<unsigned int>(Transition::EndRun), esGetTokenIndices(Transition::EndRun), false};
      this->doEndRun_(cnstR, c);
      r.setProducer(this);
      this->doEndRunProduce_(r, c);
      commit_(r);
    }

    void EDFilterBase::doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                              EventSetupImpl const& ci,
                                              ModuleCallingContext const* mcc) {
      LuminosityBlock lb(lbp, moduleDescription_, mcc, false);
      lb.setConsumer(this);
      LuminosityBlock const& cnstLb = lb;
      const EventSetup c{ci,
                         static_cast<unsigned int>(Transition::BeginLuminosityBlock),
                         esGetTokenIndices(Transition::BeginLuminosityBlock),
                         false};
      this->doBeginLuminosityBlock_(cnstLb, c);
      lb.setProducer(this);
      this->doBeginLuminosityBlockProduce_(lb, c);
      commit_(lb);
    }

    void EDFilterBase::doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                            EventSetupImpl const& ci,
                                            ModuleCallingContext const* mcc) {
      LuminosityBlock lb(lbp, moduleDescription_, mcc, true);
      lb.setConsumer(this);
      LuminosityBlock const& cnstLb = lb;
      const EventSetup c{ci,
                         static_cast<unsigned int>(Transition::EndLuminosityBlock),
                         esGetTokenIndices(Transition::EndLuminosityBlock),
                         false};
      this->doEndLuminosityBlock_(cnstLb, c);
      lb.setProducer(this);
      this->doEndLuminosityBlockProduce_(lb, c);
      commit_(lb);
    }

    void EDFilterBase::doRespondToOpenInputFile(FileBlock const& fb) {
      //respondToOpenInputFile(fb);
    }

    void EDFilterBase::doRespondToCloseInputFile(FileBlock const& fb) {
      //respondToCloseInputFile(fb);
    }

    void EDFilterBase::doBeginRun_(Run const& rp, EventSetup const& c) {}
    void EDFilterBase::doEndRun_(Run const& rp, EventSetup const& c) {}
    void EDFilterBase::doBeginLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDFilterBase::doEndLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c) {}

    void EDFilterBase::doBeginRunProduce_(Run& rp, EventSetup const& c) {}
    void EDFilterBase::doEndRunProduce_(Run& rp, EventSetup const& c) {}
    void EDFilterBase::doBeginLuminosityBlockProduce_(LuminosityBlock& lbp, EventSetup const& c) {}
    void EDFilterBase::doEndLuminosityBlockProduce_(LuminosityBlock& lbp, EventSetup const& c) {}

    void EDFilterBase::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;
      desc.setUnknown();
      descriptions.addDefault(desc);
    }

    void EDFilterBase::prevalidate(ConfigurationDescriptions& iConfig) { edmodule_mightGet_config(iConfig); }

    static const std::string kBaseType("EDFilter");

    const std::string& EDFilterBase::baseType() { return kBaseType; }

  }  // namespace one
}  // namespace edm
