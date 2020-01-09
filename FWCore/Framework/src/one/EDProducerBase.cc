// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     one::EDProducerBase
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Thu, 02 May 2013 21:56:04 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/one/EDProducerBase.h"
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

  namespace one {
    //
    // static data member definitions
    //

    //
    // constructors and destructor
    //
    EDProducerBase::EDProducerBase()
        : ProducerBase(), moduleDescription_(), previousParentage_(), previousParentageId_() {}

    EDProducerBase::~EDProducerBase() {}

    bool EDProducerBase::doEvent(EventPrincipal const& ep,
                                 EventSetupImpl const& ci,
                                 ActivityRegistry* act,
                                 ModuleCallingContext const* mcc) {
      Event e(ep, moduleDescription_, mcc);
      e.setConsumer(this);
      e.setProducer(this, &previousParentage_);
      e.setSharedResourcesAcquirer(&resourcesAcquirer_);
      EventSignalsSentry sentry(act, mcc);
      const EventSetup c{ci, static_cast<unsigned int>(Transition::Event), esGetTokenIndices(Transition::Event), false};
      this->produce(e, c);
      commit_(e, &previousParentageId_);
      return true;
    }

    SharedResourcesAcquirer EDProducerBase::createAcquirer() {
      return SharedResourcesAcquirer{
          std::vector<std::shared_ptr<SerialTaskQueue>>(1, std::make_shared<SerialTaskQueue>())};
    }

    SerialTaskQueue* EDProducerBase::globalRunsQueue() { return nullptr; }
    SerialTaskQueue* EDProducerBase::globalLuminosityBlocksQueue() { return nullptr; };

    void EDProducerBase::doBeginJob() {
      resourcesAcquirer_ = createAcquirer();

      this->beginJob();
    }

    void EDProducerBase::doEndJob() { this->endJob(); }

    void EDProducerBase::doPreallocate(PreallocationConfiguration const& iPrealloc) {
      auto const nThreads = iPrealloc.numberOfThreads();
      preallocThreads(nThreads);
      preallocLumis(iPrealloc.numberOfLuminosityBlocks());
    }

    void EDProducerBase::preallocLumis(unsigned int){};

    void EDProducerBase::doBeginRun(RunPrincipal const& rp, EventSetupImpl const& ci, ModuleCallingContext const* mcc) {
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

    void EDProducerBase::doEndRun(RunPrincipal const& rp, EventSetupImpl const& ci, ModuleCallingContext const* mcc) {
      Run r(rp, moduleDescription_, mcc, true);
      r.setConsumer(this);
      Run const& cnstR = r;
      r.setProducer(this);
      const EventSetup c{
          ci, static_cast<unsigned int>(Transition::EndRun), esGetTokenIndices(Transition::EndRun), false};
      this->doEndRunProduce_(r, c);
      this->doEndRun_(cnstR, c);
      commit_(r);
    }

    void EDProducerBase::doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp,
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

    void EDProducerBase::doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                              EventSetupImpl const& ci,
                                              ModuleCallingContext const* mcc) {
      LuminosityBlock lb(lbp, moduleDescription_, mcc, true);
      lb.setConsumer(this);
      lb.setProducer(this);
      const EventSetup c{ci,
                         static_cast<unsigned int>(Transition::EndLuminosityBlock),
                         esGetTokenIndices(Transition::EndLuminosityBlock),
                         false};
      this->doEndLuminosityBlockProduce_(lb, c);
      LuminosityBlock const& cnstLb = lb;
      this->doEndLuminosityBlock_(cnstLb, c);
      commit_(lb);
    }

    void EDProducerBase::doRespondToOpenInputFile(FileBlock const& fb) {
      //respondToOpenInputFile(fb);
    }

    void EDProducerBase::doRespondToCloseInputFile(FileBlock const& fb) {
      //respondToCloseInputFile(fb);
    }

    void EDProducerBase::doBeginRun_(Run const& rp, EventSetup const& c) {}
    void EDProducerBase::doEndRun_(Run const& rp, EventSetup const& c) {}
    void EDProducerBase::doBeginLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDProducerBase::doEndLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c) {}

    void EDProducerBase::doBeginRunProduce_(Run& rp, EventSetup const& c) {}
    void EDProducerBase::doEndRunProduce_(Run& rp, EventSetup const& c) {}
    void EDProducerBase::doBeginLuminosityBlockProduce_(LuminosityBlock& lbp, EventSetup const& c) {}
    void EDProducerBase::doEndLuminosityBlockProduce_(LuminosityBlock& lbp, EventSetup const& c) {}

    void EDProducerBase::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;
      desc.setUnknown();
      descriptions.addDefault(desc);
    }

    void EDProducerBase::prevalidate(ConfigurationDescriptions& iConfig) { edmodule_mightGet_config(iConfig); }

    static const std::string kBaseType("EDProducer");

    const std::string& EDProducerBase::baseType() { return kBaseType; }

  }  // namespace one
}  // namespace edm
