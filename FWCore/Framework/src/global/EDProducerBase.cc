// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     global::EDProducerBase
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
#include "FWCore/Framework/interface/global/EDProducerBase.h"
#include "FWCore/Framework/src/EventAcquireSignalsSentry.h"
#include "FWCore/Framework/src/EventSignalsSentry.h"
#include "FWCore/Framework/src/PreallocationConfiguration.h"
#include "FWCore/Framework/src/edmodule_mightGet_config.h"
#include "FWCore/Framework/src/TransitionInfoTypes.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

//
// constants, enums and typedefs
//
namespace edm {

  class WaitingTaskWithArenaHolder;

  namespace global {
    //
    // static data member definitions
    //

    //
    // constructors and destructor
    //
    EDProducerBase::EDProducerBase()
        : ProducerBase(),
          moduleDescription_(),
          previousParentages_(),
          gotBranchIDsFromAcquire_(),
          previousParentageIds_() {}

    EDProducerBase::~EDProducerBase() {}

    bool EDProducerBase::doEvent(EventTransitionInfo const& info,
                                 ActivityRegistry* act,
                                 ModuleCallingContext const* mcc) {
      Event e(info, moduleDescription_, mcc);
      e.setConsumer(this);
      const auto streamIndex = e.streamID().value();
      e.setProducer(
          this, &previousParentages_[streamIndex], hasAcquire() ? &gotBranchIDsFromAcquire_[streamIndex] : nullptr);
      EventSignalsSentry sentry(act, mcc);
      this->produce(
          e.streamID(),
          e,
          EventSetup{info, static_cast<unsigned int>(Transition::Event), esGetTokenIndices(Transition::Event), false});
      commit_(e, &previousParentageIds_[streamIndex]);
      return true;
    }

    void EDProducerBase::doAcquire(EventTransitionInfo const& info,
                                   ActivityRegistry* act,
                                   ModuleCallingContext const* mcc,
                                   WaitingTaskWithArenaHolder& holder) {
      Event e(info, moduleDescription_, mcc);
      e.setConsumer(this);
      const auto streamIndex = e.streamID().value();
      e.setProducerForAcquire(this, nullptr, gotBranchIDsFromAcquire_[streamIndex]);
      EventAcquireSignalsSentry sentry(act, mcc);
      const EventSetup c{
          info, static_cast<unsigned int>(Transition::Event), esGetTokenIndices(Transition::Event), false};
      this->doAcquire_(e.streamID(), e, c, holder);
    }

    void EDProducerBase::doPreallocate(PreallocationConfiguration const& iPrealloc) {
      auto const nStreams = iPrealloc.numberOfStreams();
      previousParentages_ = std::make_unique<std::vector<BranchID>[]>(nStreams);
      if (hasAcquire()) {
        gotBranchIDsFromAcquire_ = std::make_unique<std::vector<BranchID>[]>(nStreams);
      }
      previousParentageIds_ = std::make_unique<ParentageID[]>(nStreams);
      preallocStreams(nStreams);
      preallocLumis(iPrealloc.numberOfLuminosityBlocks());
      preallocLumisSummary(iPrealloc.numberOfLuminosityBlocks());
      preallocate(iPrealloc);
    }

    void EDProducerBase::doBeginJob() { this->beginJob(); }

    void EDProducerBase::doEndJob() { this->endJob(); }

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
      const EventSetup c{
          info, static_cast<unsigned int>(Transition::BeginRun), esGetTokenIndices(Transition::BeginRun), false};
      this->doBeginRun_(cnstR, c);
      this->doBeginRunSummary_(cnstR, c);
      r.setProducer(this);
      this->doBeginRunProduce_(r, c);
      commit_(r);
    }

    void EDProducerBase::doEndRun(RunTransitionInfo const& info, ModuleCallingContext const* mcc) {
      Run r(info, moduleDescription_, mcc, true);
      r.setConsumer(this);
      r.setProducer(this);
      Run const& cnstR = r;
      const EventSetup c{
          info, static_cast<unsigned int>(Transition::EndRun), esGetTokenIndices(Transition::EndRun), false};
      this->doEndRunSummary_(r, c);
      this->doEndRunProduce_(r, c);
      this->doEndRun_(cnstR, c);
      commit_(r);
    }

    void EDProducerBase::doBeginLuminosityBlock(LumiTransitionInfo const& info, ModuleCallingContext const* mcc) {
      LuminosityBlock lb(info, moduleDescription_, mcc, false);
      lb.setConsumer(this);
      LuminosityBlock const& cnstLb = lb;
      const EventSetup c{info,
                         static_cast<unsigned int>(Transition::BeginLuminosityBlock),
                         esGetTokenIndices(Transition::BeginLuminosityBlock),
                         false};
      this->doBeginLuminosityBlock_(cnstLb, c);
      this->doBeginLuminosityBlockSummary_(cnstLb, c);
      lb.setProducer(this);
      this->doBeginLuminosityBlockProduce_(lb, c);
      commit_(lb);
    }

    void EDProducerBase::doEndLuminosityBlock(LumiTransitionInfo const& info, ModuleCallingContext const* mcc) {
      LuminosityBlock lb(info, moduleDescription_, mcc, true);
      lb.setConsumer(this);
      lb.setProducer(this);
      LuminosityBlock const& cnstLb = lb;
      const EventSetup c{info,
                         static_cast<unsigned int>(Transition::EndLuminosityBlock),
                         esGetTokenIndices(Transition::EndLuminosityBlock),
                         false};
      this->doEndLuminosityBlockSummary_(cnstLb, c);
      this->doEndLuminosityBlockProduce_(lb, c);
      this->doEndLuminosityBlock_(cnstLb, c);
      commit_(lb);
    }

    void EDProducerBase::doBeginStream(StreamID id) { doBeginStream_(id); }
    void EDProducerBase::doEndStream(StreamID id) { doEndStream_(id); }
    void EDProducerBase::doStreamBeginRun(StreamID id, RunTransitionInfo const& info, ModuleCallingContext const* mcc) {
      Run r(info, moduleDescription_, mcc, false);
      r.setConsumer(this);
      this->doStreamBeginRun_(
          id,
          r,
          EventSetup{
              info, static_cast<unsigned int>(Transition::BeginRun), esGetTokenIndices(Transition::BeginRun), false});
    }
    void EDProducerBase::doStreamEndRun(StreamID id, RunTransitionInfo const& info, ModuleCallingContext const* mcc) {
      Run r(info, moduleDescription_, mcc, true);
      r.setConsumer(this);
      const EventSetup c{
          info, static_cast<unsigned int>(Transition::EndRun), esGetTokenIndices(Transition::EndRun), false};
      this->doStreamEndRun_(id, r, c);
      this->doStreamEndRunSummary_(id, r, c);
    }
    void EDProducerBase::doStreamBeginLuminosityBlock(StreamID id,
                                                      LumiTransitionInfo const& info,
                                                      ModuleCallingContext const* mcc) {
      LuminosityBlock lb(info, moduleDescription_, mcc, false);
      lb.setConsumer(this);
      this->doStreamBeginLuminosityBlock_(id,
                                          lb,
                                          EventSetup{info,
                                                     static_cast<unsigned int>(Transition::BeginLuminosityBlock),
                                                     esGetTokenIndices(Transition::BeginLuminosityBlock),
                                                     false});
    }

    void EDProducerBase::doStreamEndLuminosityBlock(StreamID id,
                                                    LumiTransitionInfo const& info,
                                                    ModuleCallingContext const* mcc) {
      LuminosityBlock lb(info, moduleDescription_, mcc, true);
      lb.setConsumer(this);
      const EventSetup c{info,
                         static_cast<unsigned int>(Transition::EndLuminosityBlock),
                         esGetTokenIndices(Transition::EndLuminosityBlock),
                         false};
      this->doStreamEndLuminosityBlock_(id, lb, c);
      this->doStreamEndLuminosityBlockSummary_(id, lb, c);
    }

    void EDProducerBase::doRespondToOpenInputFile(FileBlock const& fb) {
      //respondToOpenInputFile(fb);
    }

    void EDProducerBase::doRespondToCloseInputFile(FileBlock const& fb) {
      //respondToCloseInputFile(fb);
    }

    void EDProducerBase::preallocStreams(unsigned int) {}
    void EDProducerBase::preallocLumis(unsigned int) {}
    void EDProducerBase::preallocLumisSummary(unsigned int) {}
    void EDProducerBase::preallocate(PreallocationConfiguration const&) {}
    void EDProducerBase::doBeginStream_(StreamID id) {}
    void EDProducerBase::doEndStream_(StreamID id) {}
    void EDProducerBase::doStreamBeginRun_(StreamID id, Run const& rp, EventSetup const& c) {}
    void EDProducerBase::doStreamEndRun_(StreamID id, Run const& rp, EventSetup const& c) {}
    void EDProducerBase::doStreamEndRunSummary_(StreamID id, Run const& rp, EventSetup const& c) {}
    void EDProducerBase::doStreamBeginLuminosityBlock_(StreamID id, LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDProducerBase::doStreamEndLuminosityBlock_(StreamID id, LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDProducerBase::doStreamEndLuminosityBlockSummary_(StreamID id,
                                                            LuminosityBlock const& lbp,
                                                            EventSetup const& c) {}

    void EDProducerBase::doBeginProcessBlock_(ProcessBlock const&) {}
    void EDProducerBase::doAccessInputProcessBlock_(ProcessBlock const&) {}
    void EDProducerBase::doEndProcessBlock_(ProcessBlock const&) {}
    void EDProducerBase::doBeginRun_(Run const& rp, EventSetup const& c) {}
    void EDProducerBase::doEndRun_(Run const& rp, EventSetup const& c) {}
    void EDProducerBase::doBeginRunSummary_(Run const& rp, EventSetup const& c) {}
    void EDProducerBase::doEndRunSummary_(Run const& rp, EventSetup const& c) {}

    void EDProducerBase::doBeginLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDProducerBase::doEndLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDProducerBase::doBeginLuminosityBlockSummary_(LuminosityBlock const& rp, EventSetup const& c) {}
    void EDProducerBase::doEndLuminosityBlockSummary_(LuminosityBlock const& lb, EventSetup const& c) {}

    void EDProducerBase::doBeginProcessBlockProduce_(ProcessBlock&) {}
    void EDProducerBase::doEndProcessBlockProduce_(ProcessBlock&) {}
    void EDProducerBase::doBeginRunProduce_(Run& rp, EventSetup const& c) {}
    void EDProducerBase::doEndRunProduce_(Run& rp, EventSetup const& c) {}
    void EDProducerBase::doBeginLuminosityBlockProduce_(LuminosityBlock& lbp, EventSetup const& c) {}
    void EDProducerBase::doEndLuminosityBlockProduce_(LuminosityBlock& lbp, EventSetup const& c) {}

    void EDProducerBase::doAcquire_(StreamID, Event const&, EventSetup const&, WaitingTaskWithArenaHolder&) {}

    void EDProducerBase::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;
      desc.setUnknown();
      descriptions.addDefault(desc);
    }

    void EDProducerBase::prevalidate(ConfigurationDescriptions& iConfig) { edmodule_mightGet_config(iConfig); }

    static const std::string kBaseType("EDProducer");

    const std::string& EDProducerBase::baseType() { return kBaseType; }

  }  // namespace global
}  // namespace edm
