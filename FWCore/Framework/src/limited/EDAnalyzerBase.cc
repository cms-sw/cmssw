// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     limited::EDAnalyzerBase
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Thu, 02 May 2013 21:56:04 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/limited/EDAnalyzerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/ProcessBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/src/edmodule_mightGet_config.h"
#include "FWCore/Framework/interface/PreallocationConfiguration.h"
#include "FWCore/Framework/src/EventSignalsSentry.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Provenance/interface/ProductRegistry.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ESParentContext.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"

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
    EDAnalyzerBase::EDAnalyzerBase(ParameterSet const& pset)
        : moduleDescription_(), queue_(pset.getUntrackedParameter<unsigned int>("concurrencyLimit")) {}

    EDAnalyzerBase::~EDAnalyzerBase() {}

    bool EDAnalyzerBase::doEvent(EventTransitionInfo const& info,
                                 ActivityRegistry* act,
                                 ModuleCallingContext const* mcc) {
      Event e(info, moduleDescription_, mcc);
      e.setConsumer(this);
      EventSignalsSentry sentry(act, mcc);
      ESParentContext parentC(mcc);
      const EventSetup c{
          info, static_cast<unsigned int>(Transition::Event), esGetTokenIndices(Transition::Event), parentC};
      this->analyze(e.streamID(), e, c);
      return true;
    }

    void EDAnalyzerBase::doPreallocate(PreallocationConfiguration const& iPrealloc) {
      preallocStreams(iPrealloc.numberOfStreams());
      preallocRuns(iPrealloc.numberOfRuns());
      preallocRunsSummary(iPrealloc.numberOfRuns());
      preallocLumis(iPrealloc.numberOfLuminosityBlocks());
      preallocLumisSummary(iPrealloc.numberOfLuminosityBlocks());
      preallocate(iPrealloc);
    }

    void EDAnalyzerBase::doBeginJob() { this->beginJob(); }

    void EDAnalyzerBase::doEndJob() { this->endJob(); }

    void EDAnalyzerBase::doBeginProcessBlock(ProcessBlockPrincipal const& pbp, ModuleCallingContext const* mcc) {
      ProcessBlock processBlock(pbp, moduleDescription_, mcc, false);
      processBlock.setConsumer(this);
      ProcessBlock const& constProcessBlock = processBlock;
      this->doBeginProcessBlock_(constProcessBlock);
    }

    void EDAnalyzerBase::doAccessInputProcessBlock(ProcessBlockPrincipal const& pbp, ModuleCallingContext const* mcc) {
      ProcessBlock processBlock(pbp, moduleDescription_, mcc, false);
      processBlock.setConsumer(this);
      ProcessBlock const& constProcessBlock = processBlock;
      this->doAccessInputProcessBlock_(constProcessBlock);
    }

    void EDAnalyzerBase::doEndProcessBlock(ProcessBlockPrincipal const& pbp, ModuleCallingContext const* mcc) {
      ProcessBlock processBlock(pbp, moduleDescription_, mcc, true);
      processBlock.setConsumer(this);
      ProcessBlock const& constProcessBlock = processBlock;
      this->doEndProcessBlock_(constProcessBlock);
    }

    void EDAnalyzerBase::doBeginRun(RunTransitionInfo const& info, ModuleCallingContext const* mcc) {
      Run r(info, moduleDescription_, mcc, false);
      r.setConsumer(this);
      Run const& cnstR = r;
      ESParentContext parentC(mcc);
      const EventSetup c{
          info, static_cast<unsigned int>(Transition::BeginRun), esGetTokenIndices(Transition::BeginRun), parentC};
      this->doBeginRun_(cnstR, c);
      this->doBeginRunSummary_(cnstR, c);
    }

    void EDAnalyzerBase::doEndRun(RunTransitionInfo const& info, ModuleCallingContext const* mcc) {
      Run r(info, moduleDescription_, mcc, true);
      r.setConsumer(this);
      Run const& cnstR = r;
      ESParentContext parentC(mcc);
      const EventSetup c{
          info, static_cast<unsigned int>(Transition::EndRun), esGetTokenIndices(Transition::EndRun), parentC};
      this->doEndRunSummary_(r, c);
      this->doEndRun_(cnstR, c);
    }

    void EDAnalyzerBase::doBeginLuminosityBlock(LumiTransitionInfo const& info, ModuleCallingContext const* mcc) {
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
    }

    void EDAnalyzerBase::doEndLuminosityBlock(LumiTransitionInfo const& info, ModuleCallingContext const* mcc) {
      LuminosityBlock lb(info, moduleDescription_, mcc, true);
      lb.setConsumer(this);
      LuminosityBlock const& cnstLb = lb;
      ESParentContext parentC(mcc);
      const EventSetup c{info,
                         static_cast<unsigned int>(Transition::EndLuminosityBlock),
                         esGetTokenIndices(Transition::EndLuminosityBlock),
                         parentC};
      this->doEndLuminosityBlockSummary_(cnstLb, c);
      this->doEndLuminosityBlock_(cnstLb, c);
    }

    void EDAnalyzerBase::doBeginStream(StreamID id) { doBeginStream_(id); }
    void EDAnalyzerBase::doEndStream(StreamID id) { doEndStream_(id); }
    void EDAnalyzerBase::doStreamBeginRun(StreamID id, RunTransitionInfo const& info, ModuleCallingContext const* mcc) {
      Run r(info, moduleDescription_, mcc, false);
      r.setConsumer(this);
      ESParentContext parentC(mcc);
      const EventSetup c{
          info, static_cast<unsigned int>(Transition::BeginRun), esGetTokenIndices(Transition::BeginRun), parentC};
      this->doStreamBeginRun_(id, r, c);
    }
    void EDAnalyzerBase::doStreamEndRun(StreamID id, RunTransitionInfo const& info, ModuleCallingContext const* mcc) {
      Run r(info, moduleDescription_, mcc, true);
      r.setConsumer(this);
      ESParentContext parentC(mcc);
      const EventSetup c{
          info, static_cast<unsigned int>(Transition::EndRun), esGetTokenIndices(Transition::EndRun), parentC};
      this->doStreamEndRun_(id, r, c);
      this->doStreamEndRunSummary_(id, r, c);
    }
    void EDAnalyzerBase::doStreamBeginLuminosityBlock(StreamID id,
                                                      LumiTransitionInfo const& info,
                                                      ModuleCallingContext const* mcc) {
      LuminosityBlock lb(info, moduleDescription_, mcc, false);
      lb.setConsumer(this);
      ESParentContext parentC(mcc);
      const EventSetup c{info,
                         static_cast<unsigned int>(Transition::BeginLuminosityBlock),
                         esGetTokenIndices(Transition::BeginLuminosityBlock),
                         parentC};
      this->doStreamBeginLuminosityBlock_(id, lb, c);
    }

    void EDAnalyzerBase::doStreamEndLuminosityBlock(StreamID id,
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

    void EDAnalyzerBase::preallocStreams(unsigned int) {}
    void EDAnalyzerBase::preallocRuns(unsigned int) {}
    void EDAnalyzerBase::preallocRunsSummary(unsigned int) {}
    void EDAnalyzerBase::preallocLumis(unsigned int) {}
    void EDAnalyzerBase::preallocLumisSummary(unsigned int) {}
    void EDAnalyzerBase::preallocate(PreallocationConfiguration const&) {}
    void EDAnalyzerBase::doBeginStream_(StreamID id) {}
    void EDAnalyzerBase::doEndStream_(StreamID id) {}
    void EDAnalyzerBase::doStreamBeginRun_(StreamID id, Run const& rp, EventSetup const& c) {}
    void EDAnalyzerBase::doStreamEndRun_(StreamID id, Run const& rp, EventSetup const& c) {}
    void EDAnalyzerBase::doStreamEndRunSummary_(StreamID id, Run const& rp, EventSetup const& c) {}
    void EDAnalyzerBase::doStreamBeginLuminosityBlock_(StreamID id, LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDAnalyzerBase::doStreamEndLuminosityBlock_(StreamID id, LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDAnalyzerBase::doStreamEndLuminosityBlockSummary_(StreamID id,
                                                            LuminosityBlock const& lbp,
                                                            EventSetup const& c) {}

    void EDAnalyzerBase::doBeginProcessBlock_(ProcessBlock const&) {}
    void EDAnalyzerBase::doAccessInputProcessBlock_(ProcessBlock const&) {}
    void EDAnalyzerBase::doEndProcessBlock_(ProcessBlock const&) {}
    void EDAnalyzerBase::doBeginRun_(Run const& rp, EventSetup const& c) {}
    void EDAnalyzerBase::doEndRun_(Run const& rp, EventSetup const& c) {}
    void EDAnalyzerBase::doBeginRunSummary_(Run const& rp, EventSetup const& c) {}
    void EDAnalyzerBase::doEndRunSummary_(Run const& rp, EventSetup const& c) {}

    void EDAnalyzerBase::doBeginLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDAnalyzerBase::doEndLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDAnalyzerBase::doBeginLuminosityBlockSummary_(LuminosityBlock const& rp, EventSetup const& c) {}
    void EDAnalyzerBase::doEndLuminosityBlockSummary_(LuminosityBlock const& lb, EventSetup const& c) {}

    void EDAnalyzerBase::clearInputProcessBlockCaches() {}

    void EDAnalyzerBase::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;
      desc.setUnknown();
      descriptions.addDefault(desc);
    }

    void EDAnalyzerBase::prevalidate(ConfigurationDescriptions& iConfig) { edmodule_mightGet_config(iConfig); }

    void EDAnalyzerBase::registerProductsAndCallbacks(EDAnalyzerBase*, ProductRegistry* reg) {
      if (callWhenNewProductsRegistered_) {
        reg->callForEachBranch(callWhenNewProductsRegistered_);

        Service<ConstProductRegistry> regService;
        regService->watchProductAdditions(callWhenNewProductsRegistered_);
      }
    }
    static const std::string kBaseType("EDAnalyzer");

    const std::string& EDAnalyzerBase::baseType() { return kBaseType; }

  }  // namespace limited
}  // namespace edm
