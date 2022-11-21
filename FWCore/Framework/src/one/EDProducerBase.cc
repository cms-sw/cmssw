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
#include "FWCore/Framework/interface/ProcessBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/src/edmodule_mightGet_config.h"
#include "FWCore/Framework/interface/PreallocationConfiguration.h"
#include "FWCore/Framework/src/EventSignalsSentry.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/Framework/interface/EventForTransformer.h"
#include "FWCore/ServiceRegistry/interface/ESParentContext.h"

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

    bool EDProducerBase::doEvent(EventTransitionInfo const& info,
                                 ActivityRegistry* act,
                                 ModuleCallingContext const* mcc) {
      Event e(info, moduleDescription_, mcc);
      e.setConsumer(this);
      e.setProducer(this, &previousParentage_);
      e.setSharedResourcesAcquirer(&resourcesAcquirer_);
      EventSignalsSentry sentry(act, mcc);
      ESParentContext parentC(mcc);
      const EventSetup c{
          info, static_cast<unsigned int>(Transition::Event), esGetTokenIndices(Transition::Event), parentC};
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

    void EDProducerBase::doTransformAsync(WaitingTaskHolder iTask,
                                          size_t iTransformIndex,
                                          EventPrincipal const& iEvent,
                                          ActivityRegistry*,
                                          ModuleCallingContext const* iMCC,
                                          ServiceWeakToken const& iToken) {
      EventForTransformer ev(iEvent, iMCC);
      transformAsync_(iTask, iTransformIndex, ev, iToken);
    }

    size_t EDProducerBase::transformIndex_(edm::BranchDescription const& iBranch) const { return -1; }
    ProductResolverIndex EDProducerBase::transformPrefetch_(std::size_t iIndex) const { return 0; }
    void EDProducerBase::transformAsync_(WaitingTaskHolder iTask,
                                         std::size_t iIndex,
                                         edm::EventForTransformer& iEvent,
                                         ServiceWeakToken const& iToken) const {}

    void EDProducerBase::doBeginJob() {
      resourcesAcquirer_ = createAcquirer();

      this->beginJob();
    }

    void EDProducerBase::doEndJob() { this->endJob(); }

    void EDProducerBase::doPreallocate(PreallocationConfiguration const& iPrealloc) {
      auto const nThreads = iPrealloc.numberOfThreads();
      preallocThreads(nThreads);
      preallocRuns(iPrealloc.numberOfRuns());
      preallocLumis(iPrealloc.numberOfLuminosityBlocks());
    }

    void EDProducerBase::preallocRuns(unsigned int) {}
    void EDProducerBase::preallocLumis(unsigned int) {}

    void EDProducerBase::doBeginProcessBlock(ProcessBlockPrincipal const& pbp, ModuleCallingContext const* mcc) {
      ProcessBlock processBlock(pbp, moduleDescription_, mcc, false);
      processBlock.setConsumer(this);
      ProcessBlock const& constProcessBlock = processBlock;
      this->doBeginProcessBlock_(constProcessBlock);
      processBlock.setProducer(this);
      this->doBeginProcessBlockProduce_(processBlock);
      commit_(processBlock);
    }

    void EDProducerBase::doAccessInputProcessBlock(ProcessBlockPrincipal const& pbp, ModuleCallingContext const* mcc) {
      ProcessBlock processBlock(pbp, moduleDescription_, mcc, false);
      processBlock.setConsumer(this);
      ProcessBlock const& constProcessBlock = processBlock;
      this->doAccessInputProcessBlock_(constProcessBlock);
    }

    void EDProducerBase::doEndProcessBlock(ProcessBlockPrincipal const& pbp, ModuleCallingContext const* mcc) {
      ProcessBlock processBlock(pbp, moduleDescription_, mcc, true);
      processBlock.setConsumer(this);
      ProcessBlock const& constProcessBlock = processBlock;
      this->doEndProcessBlock_(constProcessBlock);
      processBlock.setProducer(this);
      this->doEndProcessBlockProduce_(processBlock);
      commit_(processBlock);
    }

    void EDProducerBase::doBeginRun(RunTransitionInfo const& info, ModuleCallingContext const* mcc) {
      Run r(info, moduleDescription_, mcc, false);
      r.setConsumer(this);
      Run const& cnstR = r;
      ESParentContext parentC(mcc);
      const EventSetup c{
          info, static_cast<unsigned int>(Transition::BeginRun), esGetTokenIndices(Transition::BeginRun), parentC};
      this->doBeginRun_(cnstR, c);
      r.setProducer(this);
      this->doBeginRunProduce_(r, c);
      commit_(r);
    }

    void EDProducerBase::doEndRun(RunTransitionInfo const& info, ModuleCallingContext const* mcc) {
      Run r(info, moduleDescription_, mcc, true);
      r.setConsumer(this);
      Run const& cnstR = r;
      r.setProducer(this);
      ESParentContext parentC(mcc);
      const EventSetup c{
          info, static_cast<unsigned int>(Transition::EndRun), esGetTokenIndices(Transition::EndRun), parentC};
      this->doEndRunProduce_(r, c);
      this->doEndRun_(cnstR, c);
      commit_(r);
    }

    void EDProducerBase::doBeginLuminosityBlock(LumiTransitionInfo const& info, ModuleCallingContext const* mcc) {
      LuminosityBlock lb(info, moduleDescription_, mcc, false);
      lb.setConsumer(this);
      LuminosityBlock const& cnstLb = lb;
      ESParentContext parentC(mcc);
      const EventSetup c{info,
                         static_cast<unsigned int>(Transition::BeginLuminosityBlock),
                         esGetTokenIndices(Transition::BeginLuminosityBlock),
                         parentC};
      this->doBeginLuminosityBlock_(cnstLb, c);
      lb.setProducer(this);
      this->doBeginLuminosityBlockProduce_(lb, c);
      commit_(lb);
    }

    void EDProducerBase::doEndLuminosityBlock(LumiTransitionInfo const& info, ModuleCallingContext const* mcc) {
      LuminosityBlock lb(info, moduleDescription_, mcc, true);
      lb.setConsumer(this);
      lb.setProducer(this);
      ESParentContext parentC(mcc);
      const EventSetup c{info,
                         static_cast<unsigned int>(Transition::EndLuminosityBlock),
                         esGetTokenIndices(Transition::EndLuminosityBlock),
                         parentC};
      this->doEndLuminosityBlockProduce_(lb, c);
      LuminosityBlock const& cnstLb = lb;
      this->doEndLuminosityBlock_(cnstLb, c);
      commit_(lb);
    }

    void EDProducerBase::doBeginProcessBlock_(ProcessBlock const&) {}
    void EDProducerBase::doAccessInputProcessBlock_(ProcessBlock const&) {}
    void EDProducerBase::doEndProcessBlock_(ProcessBlock const&) {}
    void EDProducerBase::doBeginRun_(Run const& rp, EventSetup const& c) {}
    void EDProducerBase::doEndRun_(Run const& rp, EventSetup const& c) {}
    void EDProducerBase::doBeginLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDProducerBase::doEndLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c) {}

    void EDProducerBase::doBeginProcessBlockProduce_(ProcessBlock&) {}
    void EDProducerBase::doEndProcessBlockProduce_(ProcessBlock&) {}
    void EDProducerBase::doBeginRunProduce_(Run& rp, EventSetup const& c) {}
    void EDProducerBase::doEndRunProduce_(Run& rp, EventSetup const& c) {}
    void EDProducerBase::doBeginLuminosityBlockProduce_(LuminosityBlock& lbp, EventSetup const& c) {}
    void EDProducerBase::doEndLuminosityBlockProduce_(LuminosityBlock& lbp, EventSetup const& c) {}

    void EDProducerBase::clearInputProcessBlockCaches() {}

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
