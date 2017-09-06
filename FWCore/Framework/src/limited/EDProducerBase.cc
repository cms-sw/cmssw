// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     limited::EDProducerBase
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Thu, 02 May 2013 21:56:04 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/limited/EDProducerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/src/edmodule_mightGet_config.h"
#include "FWCore/Framework/src/PreallocationConfiguration.h"
#include "FWCore/Framework/src/EventSignalsSentry.h"

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
    EDProducerBase::EDProducerBase(ParameterSet const& pset):
    ProducerBase(),
    moduleDescription_(),
    previousParentages_(),
    previousParentageIds_(),
    queue_(pset.getUntrackedParameter<unsigned int>("concurrencyLimit")){ }
    
    EDProducerBase::~EDProducerBase()
    {
    }
    
    bool
    EDProducerBase::doEvent(EventPrincipal const& ep, EventSetup const& c,
                            ActivityRegistry* act,
                            ModuleCallingContext const* mcc) {
      Event e(ep, moduleDescription_, mcc);
      e.setConsumer(this);
      EventSignalsSentry sentry(act,mcc);
      this->produce(e.streamID(), e, c);
      const auto streamIndex = e.streamID().value();
      commit_(e,&previousParentages_[streamIndex], &previousParentageIds_[streamIndex]);
      return true;
    }

    void
    EDProducerBase::doPreallocate(PreallocationConfiguration const& iPrealloc) {
      auto const nStreams = iPrealloc.numberOfStreams();
      previousParentages_.reset(new std::vector<BranchID>[nStreams]);
      previousParentageIds_.reset( new ParentageID[nStreams]);
      preallocStreams(nStreams);
    }
    
    void
    EDProducerBase::doBeginJob() {
      this->beginJob();
    }
    
    void
    EDProducerBase::doEndJob() {
      this->endJob();
    }
    
    void
    EDProducerBase::doBeginRun(RunPrincipal const& rp, EventSetup const& c,
                               ModuleCallingContext const* mcc) {
      Run r(rp, moduleDescription_, mcc);
      r.setConsumer(this);
      Run const& cnstR = r;
      this->doBeginRun_(cnstR, c);
      this->doBeginRunSummary_(cnstR, c);
      this->doBeginRunProduce_(r,c);
      commit_(r);
    }
    
    void
    EDProducerBase::doEndRun(RunPrincipal const& rp, EventSetup const& c,
                             ModuleCallingContext const* mcc) {
      Run r(rp, moduleDescription_, mcc);
      r.setConsumer(this);
      Run const& cnstR = r;
      this->doEndRunProduce_(r, c);
      this->doEndRunSummary_(r,c);
      this->doEndRun_(cnstR, c);
      commit_(r);
    }
    
    void
    EDProducerBase::doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                                           ModuleCallingContext const* mcc) {
      LuminosityBlock lb(lbp, moduleDescription_, mcc);
      lb.setConsumer(this);
      LuminosityBlock const& cnstLb = lb;
      this->doBeginLuminosityBlock_(cnstLb, c);
      this->doBeginLuminosityBlockSummary_(cnstLb, c);
      this->doBeginLuminosityBlockProduce_(lb, c);
      commit_(lb);
    }
    
    void
    EDProducerBase::doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                                         ModuleCallingContext const* mcc) {
      LuminosityBlock lb(lbp, moduleDescription_, mcc);
      lb.setConsumer(this);
      LuminosityBlock const& cnstLb = lb;
      this->doEndLuminosityBlockProduce_(lb, c);
      this->doEndLuminosityBlockSummary_(cnstLb,c);
      this->doEndLuminosityBlock_(cnstLb, c);
      commit_(lb);
    }
    
    void
    EDProducerBase::doBeginStream(StreamID id) {
      doBeginStream_(id);
    }
    void
    EDProducerBase::doEndStream(StreamID id) {
      doEndStream_(id);
    }
    void
    EDProducerBase::doStreamBeginRun(StreamID id,
                                     RunPrincipal const& rp,
                                     EventSetup const& c,
                                     ModuleCallingContext const* mcc)
    {
      Run r(rp, moduleDescription_, mcc);
      r.setConsumer(this);
      this->doStreamBeginRun_(id, r, c);
    }
    void
    EDProducerBase::doStreamEndRun(StreamID id,
                                   RunPrincipal const& rp,
                                   EventSetup const& c,
                                   ModuleCallingContext const* mcc) {
      Run r(rp, moduleDescription_, mcc);
      r.setConsumer(this);
      this->doStreamEndRun_(id, r, c);
      this->doStreamEndRunSummary_(id, r, c);
    }
    void
    EDProducerBase::doStreamBeginLuminosityBlock(StreamID id,
                                                 LuminosityBlockPrincipal const& lbp,
                                                 EventSetup const& c,
                                                 ModuleCallingContext const* mcc) {
      LuminosityBlock lb(lbp, moduleDescription_, mcc);
      lb.setConsumer(this);
      this->doStreamBeginLuminosityBlock_(id,lb, c);
    }
    
    void
    EDProducerBase::doStreamEndLuminosityBlock(StreamID id,
                                               LuminosityBlockPrincipal const& lbp,
                                               EventSetup const& c,
                                               ModuleCallingContext const* mcc) {
      LuminosityBlock lb(lbp, moduleDescription_, mcc);
      lb.setConsumer(this);
      this->doStreamEndLuminosityBlock_(id,lb, c);
      this->doStreamEndLuminosityBlockSummary_(id,lb, c);
    }
    
    
    
    void
    EDProducerBase::doRespondToOpenInputFile(FileBlock const& fb) {
      //respondToOpenInputFile(fb);
    }
    
    void
    EDProducerBase::doRespondToCloseInputFile(FileBlock const& fb) {
      //respondToCloseInputFile(fb);
    }
    
    void EDProducerBase::preallocStreams(unsigned int) {}
    void EDProducerBase::doBeginStream_(StreamID id){}
    void EDProducerBase::doEndStream_(StreamID id) {}
    void EDProducerBase::doStreamBeginRun_(StreamID id, Run const& rp, EventSetup const& c) {}
    void EDProducerBase::doStreamEndRun_(StreamID id, Run const& rp, EventSetup const& c) {}
    void EDProducerBase::doStreamEndRunSummary_(StreamID id, Run const& rp, EventSetup const& c) {}
    void EDProducerBase::doStreamBeginLuminosityBlock_(StreamID id, LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDProducerBase::doStreamEndLuminosityBlock_(StreamID id, LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDProducerBase::doStreamEndLuminosityBlockSummary_(StreamID id, LuminosityBlock const& lbp, EventSetup const& c) {}
    
    
    void EDProducerBase::doBeginRun_(Run const& rp, EventSetup const& c) {}
    void EDProducerBase::doEndRun_(Run const& rp, EventSetup const& c) {}
    void EDProducerBase::doBeginRunSummary_(Run const& rp, EventSetup const& c) {}
    void EDProducerBase::doEndRunSummary_(Run const& rp, EventSetup const& c) {}
    
    void EDProducerBase::doBeginLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDProducerBase::doEndLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDProducerBase::doBeginLuminosityBlockSummary_(LuminosityBlock const& rp, EventSetup const& c) {}
    void EDProducerBase::doEndLuminosityBlockSummary_(LuminosityBlock const& lb, EventSetup const& c) {}
    
    void EDProducerBase::doBeginRunProduce_(Run& rp, EventSetup const& c) {}
    void EDProducerBase::doEndRunProduce_(Run& rp, EventSetup const& c) {}
    void EDProducerBase::doBeginLuminosityBlockProduce_(LuminosityBlock& lbp, EventSetup const& c) {}
    void EDProducerBase::doEndLuminosityBlockProduce_(LuminosityBlock& lbp, EventSetup const& c) {}
    
    void
    EDProducerBase::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;
      desc.setUnknown();
      descriptions.addDefault(desc);
    }
    
    void
    EDProducerBase::prevalidate(ConfigurationDescriptions& iConfig) {
      edmodule_mightGet_config(iConfig);
    }
    
    static const std::string kBaseType("EDProducer");
    
    const std::string&
    EDProducerBase::baseType() {
      return kBaseType;
    }

  }
}
