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
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/src/edmodule_mightGet_config.h"
#include "FWCore/Framework/src/PreallocationConfiguration.h"
#include "FWCore/Framework/src/EventSignalsSentry.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Provenance/interface/ProductRegistry.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
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

    bool EDAnalyzerBase::doEvent(EventPrincipal const& ep,
                                 EventSetupImpl const& ci,
                                 ActivityRegistry* act,
                                 ModuleCallingContext const* mcc) {
      Event e(ep, moduleDescription_, mcc);
      e.setConsumer(this);
      EventSignalsSentry sentry(act, mcc);
      const EventSetup c{ci, static_cast<unsigned int>(Transition::Event), esGetTokenIndices(Transition::Event), false};
      this->analyze(e.streamID(), e, c);
      return true;
    }

    void EDAnalyzerBase::doPreallocate(PreallocationConfiguration const& iPrealloc) {
      preallocStreams(iPrealloc.numberOfStreams());
      preallocLumis(iPrealloc.numberOfLuminosityBlocks());
      preallocLumisSummary(iPrealloc.numberOfLuminosityBlocks());
      preallocate(iPrealloc);
    }

    void EDAnalyzerBase::doBeginJob() { this->beginJob(); }

    void EDAnalyzerBase::doEndJob() { this->endJob(); }

    void EDAnalyzerBase::doBeginRun(RunPrincipal const& rp, EventSetupImpl const& ci, ModuleCallingContext const* mcc) {
      Run r(rp, moduleDescription_, mcc, false);
      r.setConsumer(this);
      Run const& cnstR = r;
      const EventSetup c{
          ci, static_cast<unsigned int>(Transition::BeginRun), esGetTokenIndices(Transition::BeginRun), false};
      this->doBeginRun_(cnstR, c);
      this->doBeginRunSummary_(cnstR, c);
    }

    void EDAnalyzerBase::doEndRun(RunPrincipal const& rp, EventSetupImpl const& ci, ModuleCallingContext const* mcc) {
      Run r(rp, moduleDescription_, mcc, true);
      r.setConsumer(this);
      Run const& cnstR = r;
      const EventSetup c{
          ci, static_cast<unsigned int>(Transition::EndRun), esGetTokenIndices(Transition::EndRun), false};
      this->doEndRunSummary_(r, c);
      this->doEndRun_(cnstR, c);
    }

    void EDAnalyzerBase::doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp,
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
      this->doBeginLuminosityBlockSummary_(cnstLb, c);
    }

    void EDAnalyzerBase::doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                              EventSetupImpl const& ci,
                                              ModuleCallingContext const* mcc) {
      LuminosityBlock lb(lbp, moduleDescription_, mcc, true);
      lb.setConsumer(this);
      LuminosityBlock const& cnstLb = lb;
      const EventSetup c{ci,
                         static_cast<unsigned int>(Transition::EndLuminosityBlock),
                         esGetTokenIndices(Transition::EndLuminosityBlock),
                         false};
      this->doEndLuminosityBlockSummary_(cnstLb, c);
      this->doEndLuminosityBlock_(cnstLb, c);
    }

    void EDAnalyzerBase::doBeginStream(StreamID id) { doBeginStream_(id); }
    void EDAnalyzerBase::doEndStream(StreamID id) { doEndStream_(id); }
    void EDAnalyzerBase::doStreamBeginRun(StreamID id,
                                          RunPrincipal const& rp,
                                          EventSetupImpl const& ci,
                                          ModuleCallingContext const* mcc) {
      Run r(rp, moduleDescription_, mcc, false);
      r.setConsumer(this);
      const EventSetup c{
          ci, static_cast<unsigned int>(Transition::BeginRun), esGetTokenIndices(Transition::BeginRun), false};
      this->doStreamBeginRun_(id, r, c);
    }
    void EDAnalyzerBase::doStreamEndRun(StreamID id,
                                        RunPrincipal const& rp,
                                        EventSetupImpl const& ci,
                                        ModuleCallingContext const* mcc) {
      Run r(rp, moduleDescription_, mcc, true);
      r.setConsumer(this);
      const EventSetup c{
          ci, static_cast<unsigned int>(Transition::EndRun), esGetTokenIndices(Transition::EndRun), false};
      this->doStreamEndRun_(id, r, c);
      this->doStreamEndRunSummary_(id, r, c);
    }
    void EDAnalyzerBase::doStreamBeginLuminosityBlock(StreamID id,
                                                      LuminosityBlockPrincipal const& lbp,
                                                      EventSetupImpl const& ci,
                                                      ModuleCallingContext const* mcc) {
      LuminosityBlock lb(lbp, moduleDescription_, mcc, false);
      lb.setConsumer(this);
      const EventSetup c{ci,
                         static_cast<unsigned int>(Transition::BeginLuminosityBlock),
                         esGetTokenIndices(Transition::BeginLuminosityBlock),
                         false};
      this->doStreamBeginLuminosityBlock_(id, lb, c);
    }

    void EDAnalyzerBase::doStreamEndLuminosityBlock(StreamID id,
                                                    LuminosityBlockPrincipal const& lbp,
                                                    EventSetupImpl const& ci,
                                                    ModuleCallingContext const* mcc) {
      LuminosityBlock lb(lbp, moduleDescription_, mcc, true);
      lb.setConsumer(this);
      const EventSetup c{ci,
                         static_cast<unsigned int>(Transition::EndLuminosityBlock),
                         esGetTokenIndices(Transition::EndLuminosityBlock),
                         false};
      this->doStreamEndLuminosityBlock_(id, lb, c);
      this->doStreamEndLuminosityBlockSummary_(id, lb, c);
    }

    void EDAnalyzerBase::doRespondToOpenInputFile(FileBlock const& fb) {
      //respondToOpenInputFile(fb);
    }

    void EDAnalyzerBase::doRespondToCloseInputFile(FileBlock const& fb) {
      //respondToCloseInputFile(fb);
    }

    void EDAnalyzerBase::preallocStreams(unsigned int) {}
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

    void EDAnalyzerBase::doBeginRun_(Run const& rp, EventSetup const& c) {}
    void EDAnalyzerBase::doEndRun_(Run const& rp, EventSetup const& c) {}
    void EDAnalyzerBase::doBeginRunSummary_(Run const& rp, EventSetup const& c) {}
    void EDAnalyzerBase::doEndRunSummary_(Run const& rp, EventSetup const& c) {}

    void EDAnalyzerBase::doBeginLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDAnalyzerBase::doEndLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDAnalyzerBase::doBeginLuminosityBlockSummary_(LuminosityBlock const& rp, EventSetup const& c) {}
    void EDAnalyzerBase::doEndLuminosityBlockSummary_(LuminosityBlock const& lb, EventSetup const& c) {}

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
