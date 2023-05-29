// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     limited::EDFilterBase
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Thu, 02 May 2013 21:56:04 GMT
//

// system include files

// user include files
#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/ProcessBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/limited/EDFilterBase.h"
#include "FWCore/Framework/src/EventSignalsSentry.h"
#include "FWCore/Framework/interface/PreallocationConfiguration.h"
#include "FWCore/Framework/src/edmodule_mightGet_config.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/Framework/interface/EventForTransformer.h"
#include "FWCore/ServiceRegistry/interface/ESParentContext.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

//
// constants, enums and typedefs
//
namespace edm {
  namespace limited {
    //
    // static data member definitions
    //

    //
    // constructors and destructor
    //
    EDFilterBase::EDFilterBase(ParameterSet const& pset)
        : ProducerBase(),
          moduleDescription_(),
          previousParentages_(),
          previousParentageIds_(),
          queue_(pset.getUntrackedParameter<unsigned int>("concurrencyLimit")) {}

    EDFilterBase::~EDFilterBase() {}

    bool EDFilterBase::doEvent(EventTransitionInfo const& info,
                               ActivityRegistry* act,
                               ModuleCallingContext const* mcc) {
      Event e(info, moduleDescription_, mcc);
      e.setConsumer(this);
      const auto streamIndex = e.streamID().value();
      e.setProducer(this, &previousParentages_[streamIndex]);
      ESParentContext parentC(mcc);
      const EventSetup c{
          info, static_cast<unsigned int>(Transition::Event), esGetTokenIndices(Transition::Event), parentC};
      EventSignalsSentry sentry(act, mcc);
      bool returnValue = this->filter(e.streamID(), e, c);
      commit_(e, &previousParentageIds_[streamIndex]);
      return returnValue;
    }

    void EDFilterBase::doTransformAsync(WaitingTaskHolder iTask,
                                        size_t iTransformIndex,
                                        EventPrincipal const& iEvent,
                                        ActivityRegistry*,
                                        ModuleCallingContext const* iMCC,
                                        ServiceWeakToken const& iToken) {
      EventForTransformer ev(iEvent, iMCC);
      transformAsync_(iTask, iTransformIndex, ev, iToken);
    }

    size_t EDFilterBase::transformIndex_(edm::BranchDescription const& iBranch) const { return -1; }
    ProductResolverIndex EDFilterBase::transformPrefetch_(std::size_t iIndex) const { return 0; }
    void EDFilterBase::transformAsync_(WaitingTaskHolder iTask,
                                       std::size_t iIndex,
                                       edm::EventForTransformer& iEvent,
                                       ServiceWeakToken const& iToken) const {}

    void EDFilterBase::doPreallocate(PreallocationConfiguration const& iPrealloc) {
      const auto nStreams = iPrealloc.numberOfStreams();
      previousParentages_ = std::make_unique<std::vector<BranchID>[]>(nStreams);
      previousParentageIds_ = std::make_unique<ParentageID[]>(nStreams);
      preallocStreams(nStreams);
      preallocRuns(iPrealloc.numberOfRuns());
      preallocRunsSummary(iPrealloc.numberOfRuns());
      preallocLumis(iPrealloc.numberOfLuminosityBlocks());
      preallocLumisSummary(iPrealloc.numberOfLuminosityBlocks());
      preallocate(iPrealloc);
    }

    void EDFilterBase::doBeginJob() { this->beginJob(); }

    void EDFilterBase::doEndJob() { this->endJob(); }

    void EDFilterBase::doBeginProcessBlock(ProcessBlockPrincipal const& pbp, ModuleCallingContext const* mcc) {
      ProcessBlock processBlock(pbp, moduleDescription_, mcc, false);
      processBlock.setConsumer(this);
      ProcessBlock const& constProcessBlock = processBlock;
      this->doBeginProcessBlock_(constProcessBlock);
      processBlock.setProducer(this);
      this->doBeginProcessBlockProduce_(processBlock);
      commit_(processBlock);
    }

    void EDFilterBase::doAccessInputProcessBlock(ProcessBlockPrincipal const& pbp, ModuleCallingContext const* mcc) {
      ProcessBlock processBlock(pbp, moduleDescription_, mcc, false);
      processBlock.setConsumer(this);
      ProcessBlock const& constProcessBlock = processBlock;
      this->doAccessInputProcessBlock_(constProcessBlock);
    }

    void EDFilterBase::doEndProcessBlock(ProcessBlockPrincipal const& pbp, ModuleCallingContext const* mcc) {
      ProcessBlock processBlock(pbp, moduleDescription_, mcc, true);
      processBlock.setConsumer(this);
      ProcessBlock const& constProcessBlock = processBlock;
      this->doEndProcessBlock_(constProcessBlock);
      processBlock.setProducer(this);
      this->doEndProcessBlockProduce_(processBlock);
      commit_(processBlock);
    }

    void EDFilterBase::doBeginRun(RunTransitionInfo const& info, ModuleCallingContext const* mcc) {
      Run r(info, moduleDescription_, mcc, false);
      r.setConsumer(this);
      Run const& cnstR = r;
      ESParentContext parentC(mcc);
      const EventSetup c{
          info, static_cast<unsigned int>(Transition::BeginRun), esGetTokenIndices(Transition::BeginRun), parentC};
      this->doBeginRun_(cnstR, c);
      this->doBeginRunSummary_(cnstR, c);
      r.setProducer(this);
      this->doBeginRunProduce_(r, c);
      commit_(r);
    }

    void EDFilterBase::doEndRun(RunTransitionInfo const& info, ModuleCallingContext const* mcc) {
      Run r(info, moduleDescription_, mcc, true);
      r.setConsumer(this);
      r.setProducer(this);
      Run const& cnstR = r;
      ESParentContext parentC(mcc);
      const EventSetup c{
          info, static_cast<unsigned int>(Transition::EndRun), esGetTokenIndices(Transition::EndRun), parentC};
      this->doEndRunSummary_(r, c);
      this->doEndRunProduce_(r, c);
      this->doEndRun_(cnstR, c);
      commit_(r);
    }

    void EDFilterBase::doBeginLuminosityBlock(LumiTransitionInfo const& info, ModuleCallingContext const* mcc) {
      LuminosityBlock lb(info, moduleDescription_, mcc, false);
      lb.setConsumer(this);
      LuminosityBlock const& cnstLb = lb;
      ESParentContext parentC(mcc);
      const EventSetup c{info,
                         static_cast<unsigned int>(Transition::BeginLuminosityBlock),
                         esGetTokenIndices(Transition::BeginLuminosityBlock),
                         parentC};
      this->doBeginLuminosityBlock_(cnstLb, c);
      this->doBeginLuminosityBlockSummary_(cnstLb, c);
      lb.setProducer(this);
      this->doBeginLuminosityBlockProduce_(lb, c);
      commit_(lb);
    }

    void EDFilterBase::doEndLuminosityBlock(LumiTransitionInfo const& info, ModuleCallingContext const* mcc) {
      LuminosityBlock lb(info, moduleDescription_, mcc, true);
      lb.setConsumer(this);
      lb.setProducer(this);
      LuminosityBlock const& cnstLb = lb;
      ESParentContext parentC(mcc);
      const EventSetup c{info,
                         static_cast<unsigned int>(Transition::EndLuminosityBlock),
                         esGetTokenIndices(Transition::EndLuminosityBlock),
                         parentC};
      this->doEndLuminosityBlockSummary_(cnstLb, c);
      this->doEndLuminosityBlockProduce_(lb, c);
      this->doEndLuminosityBlock_(cnstLb, c);
      commit_(lb);
    }

    void EDFilterBase::doBeginStream(StreamID id) { doBeginStream_(id); }
    void EDFilterBase::doEndStream(StreamID id) { doEndStream_(id); }
    void EDFilterBase::doStreamBeginRun(StreamID id, RunTransitionInfo const& info, ModuleCallingContext const* mcc) {
      Run r(info, moduleDescription_, mcc, false);
      r.setConsumer(this);
      ESParentContext parentC(mcc);
      const EventSetup c{
          info, static_cast<unsigned int>(Transition::BeginRun), esGetTokenIndices(Transition::BeginRun), parentC};
      this->doStreamBeginRun_(id, r, c);
    }
    void EDFilterBase::doStreamEndRun(StreamID id, RunTransitionInfo const& info, ModuleCallingContext const* mcc) {
      Run r(info, moduleDescription_, mcc, true);
      r.setConsumer(this);
      ESParentContext parentC(mcc);
      const EventSetup c{
          info, static_cast<unsigned int>(Transition::EndRun), esGetTokenIndices(Transition::EndRun), parentC};
      this->doStreamEndRun_(id, r, c);
      this->doStreamEndRunSummary_(id, r, c);
    }
    void EDFilterBase::doStreamBeginLuminosityBlock(StreamID id,
                                                    LumiTransitionInfo const& info,
                                                    ModuleCallingContext const* mcc) {
      LuminosityBlock lb(info, moduleDescription_, mcc, false);
      ESParentContext parentC(mcc);
      const EventSetup c{info,
                         static_cast<unsigned int>(Transition::BeginLuminosityBlock),
                         esGetTokenIndices(Transition::BeginLuminosityBlock),
                         parentC};
      lb.setConsumer(this);
      this->doStreamBeginLuminosityBlock_(id, lb, c);
    }

    void EDFilterBase::doStreamEndLuminosityBlock(StreamID id,
                                                  LumiTransitionInfo const& info,
                                                  ModuleCallingContext const* mcc) {
      LuminosityBlock lb(info, moduleDescription_, mcc, true);
      lb.setConsumer(this);
      ESParentContext parentC(mcc);
      const EventSetup c{info,
                         static_cast<unsigned int>(Transition::EndLuminosityBlock),
                         esGetTokenIndices(Transition::EndLuminosityBlock),
                         parentC};
      this->doStreamEndLuminosityBlock_(id, lb, c);
      this->doStreamEndLuminosityBlockSummary_(id, lb, c);
    }

    void EDFilterBase::preallocStreams(unsigned int) {}
    void EDFilterBase::preallocRuns(unsigned int) {}
    void EDFilterBase::preallocRunsSummary(unsigned int) {}
    void EDFilterBase::preallocLumis(unsigned int) {}
    void EDFilterBase::preallocLumisSummary(unsigned int) {}
    void EDFilterBase::preallocate(PreallocationConfiguration const&) {}
    void EDFilterBase::doBeginStream_(StreamID id) {}
    void EDFilterBase::doEndStream_(StreamID id) {}
    void EDFilterBase::doStreamBeginRun_(StreamID id, Run const& rp, EventSetup const& c) {}
    void EDFilterBase::doStreamEndRun_(StreamID id, Run const& rp, EventSetup const& c) {}
    void EDFilterBase::doStreamEndRunSummary_(StreamID id, Run const& rp, EventSetup const& c) {}
    void EDFilterBase::doStreamBeginLuminosityBlock_(StreamID id, LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDFilterBase::doStreamEndLuminosityBlock_(StreamID id, LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDFilterBase::doStreamEndLuminosityBlockSummary_(StreamID id,
                                                          LuminosityBlock const& lbp,
                                                          EventSetup const& c) {}

    void EDFilterBase::doBeginProcessBlock_(ProcessBlock const&) {}
    void EDFilterBase::doAccessInputProcessBlock_(ProcessBlock const&) {}
    void EDFilterBase::doEndProcessBlock_(ProcessBlock const&) {}
    void EDFilterBase::doBeginRun_(Run const& rp, EventSetup const& c) {}
    void EDFilterBase::doEndRun_(Run const& rp, EventSetup const& c) {}
    void EDFilterBase::doBeginRunSummary_(Run const& rp, EventSetup const& c) {}
    void EDFilterBase::doEndRunSummary_(Run const& rp, EventSetup const& c) {}

    void EDFilterBase::doBeginLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDFilterBase::doEndLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDFilterBase::doBeginLuminosityBlockSummary_(LuminosityBlock const& rp, EventSetup const& c) {}
    void EDFilterBase::doEndLuminosityBlockSummary_(LuminosityBlock const& lb, EventSetup const& c) {}

    void EDFilterBase::doBeginProcessBlockProduce_(ProcessBlock&) {}
    void EDFilterBase::doEndProcessBlockProduce_(ProcessBlock&) {}
    void EDFilterBase::doBeginRunProduce_(Run& rp, EventSetup const& c) {}
    void EDFilterBase::doEndRunProduce_(Run& rp, EventSetup const& c) {}
    void EDFilterBase::doBeginLuminosityBlockProduce_(LuminosityBlock& lbp, EventSetup const& c) {}
    void EDFilterBase::doEndLuminosityBlockProduce_(LuminosityBlock& lbp, EventSetup const& c) {}

    void EDFilterBase::clearInputProcessBlockCaches() {}

    void EDFilterBase::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;
      desc.setUnknown();
      descriptions.addDefault(desc);
    }

    void EDFilterBase::prevalidate(ConfigurationDescriptions& iConfig) { edmodule_mightGet_config(iConfig); }

    static const std::string kBaseType("EDFilter");

    const std::string& EDFilterBase::baseType() { return kBaseType; }

  }  // namespace limited
}  // namespace edm
