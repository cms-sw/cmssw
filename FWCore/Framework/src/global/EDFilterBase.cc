// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     global::EDFilterBase
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Thu, 02 May 2013 21:56:04 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/global/EDFilterBase.h"
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
  namespace global {
    //
    // static data member definitions
    //
    
    //
    // constructors and destructor
    //
    EDFilterBase::EDFilterBase():
    ProducerBase(),
    moduleDescription_(),
    previousParentages_(),
    previousParentageIds_() { }
    
    EDFilterBase::~EDFilterBase()
    {
    }
    
    bool
    EDFilterBase::doEvent(EventPrincipal const& ep, EventSetup const& c,
                          ActivityRegistry* act,
                          ModuleCallingContext const* mcc) {
      Event e(ep, moduleDescription_, mcc);
      e.setConsumer(this);
      EventSignalsSentry sentry(act,mcc);
      bool returnValue = this->filter(e.streamID(), e, c);
      const auto streamIndex =e.streamID().value();
      commit_(e,&previousParentages_[streamIndex], &previousParentageIds_[streamIndex]);
      return returnValue;
    }
    
    void
    EDFilterBase::doPreallocate(PreallocationConfiguration const& iPrealloc) {
      const auto nStreams =iPrealloc.numberOfStreams();
      previousParentages_.reset(new std::vector<BranchID>[nStreams]);
      previousParentageIds_.reset(new ParentageID[nStreams]);
      preallocStreams(nStreams);
    }

    void
    EDFilterBase::doBeginJob() {
      this->beginJob();
    }
    
    void
    EDFilterBase::doEndJob() {
      this->endJob();
    }
    
    void
    EDFilterBase::doBeginRun(RunPrincipal const& rp, EventSetup const& c,
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
    EDFilterBase::doEndRun(RunPrincipal const& rp, EventSetup const& c,
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
    EDFilterBase::doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
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
    EDFilterBase::doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
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
    EDFilterBase::doBeginStream(StreamID id) {
      doBeginStream_(id);
    }
    void
    EDFilterBase::doEndStream(StreamID id) {
      doEndStream_(id);
    }
    void
    EDFilterBase::doStreamBeginRun(StreamID id,
                                     RunPrincipal const& rp,
                                     EventSetup const& c,
                                     ModuleCallingContext const* mcc)
    {
      Run r(rp, moduleDescription_, mcc);
      r.setConsumer(this);
      this->doStreamBeginRun_(id, r, c);
    }
    void
    EDFilterBase::doStreamEndRun(StreamID id,
                                   RunPrincipal const& rp,
                                   EventSetup const& c,
                                   ModuleCallingContext const* mcc) {
      Run r(rp, moduleDescription_, mcc);
      r.setConsumer(this);
      this->doStreamEndRun_(id, r, c);
      this->doStreamEndRunSummary_(id, r, c);
    }
    void
    EDFilterBase::doStreamBeginLuminosityBlock(StreamID id,
                                                 LuminosityBlockPrincipal const& lbp,
                                                 EventSetup const& c,
                                                 ModuleCallingContext const* mcc) {
      LuminosityBlock lb(lbp, moduleDescription_, mcc);
      lb.setConsumer(this);
      this->doStreamBeginLuminosityBlock_(id,lb, c);
    }
    
    void
    EDFilterBase::doStreamEndLuminosityBlock(StreamID id,
                                               LuminosityBlockPrincipal const& lbp,
                                               EventSetup const& c,
                                               ModuleCallingContext const* mcc) {
      LuminosityBlock lb(lbp, moduleDescription_, mcc);
      lb.setConsumer(this);
      this->doStreamEndLuminosityBlock_(id,lb, c);
      this->doStreamEndLuminosityBlockSummary_(id,lb, c);
    }
    
    
    
    void
    EDFilterBase::doRespondToOpenInputFile(FileBlock const& fb) {
      //respondToOpenInputFile(fb);
    }
    
    void
    EDFilterBase::doRespondToCloseInputFile(FileBlock const& fb) {
      //respondToCloseInputFile(fb);
    }
    
    void
    EDFilterBase::doPreForkReleaseResources() {
      preForkReleaseResources();
    }
    
    void
    EDFilterBase::doPostForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren) {
      postForkReacquireResources(iChildIndex, iNumberOfChildren);
    }
    
    void EDFilterBase::preallocStreams(unsigned int) {}
    void EDFilterBase::doBeginStream_(StreamID id){}
    void EDFilterBase::doEndStream_(StreamID id) {}
    void EDFilterBase::doStreamBeginRun_(StreamID id, Run const& rp, EventSetup const& c) {}
    void EDFilterBase::doStreamEndRun_(StreamID id, Run const& rp, EventSetup const& c) {}
    void EDFilterBase::doStreamEndRunSummary_(StreamID id, Run const& rp, EventSetup const& c) {}
    void EDFilterBase::doStreamBeginLuminosityBlock_(StreamID id, LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDFilterBase::doStreamEndLuminosityBlock_(StreamID id, LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDFilterBase::doStreamEndLuminosityBlockSummary_(StreamID id, LuminosityBlock const& lbp, EventSetup const& c) {}
    
    
    void EDFilterBase::doBeginRun_(Run const& rp, EventSetup const& c) {}
    void EDFilterBase::doEndRun_(Run const& rp, EventSetup const& c) {}
    void EDFilterBase::doBeginRunSummary_(Run const& rp, EventSetup const& c) {}
    void EDFilterBase::doEndRunSummary_(Run const& rp, EventSetup const& c) {}
    
    void EDFilterBase::doBeginLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDFilterBase::doEndLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDFilterBase::doBeginLuminosityBlockSummary_(LuminosityBlock const& rp, EventSetup const& c) {}
    void EDFilterBase::doEndLuminosityBlockSummary_(LuminosityBlock const& lb, EventSetup const& c) {}
    
    void EDFilterBase::doBeginRunProduce_(Run& rp, EventSetup const& c) {}
    void EDFilterBase::doEndRunProduce_(Run& rp, EventSetup const& c) {}
    void EDFilterBase::doBeginLuminosityBlockProduce_(LuminosityBlock& lbp, EventSetup const& c) {}
    void EDFilterBase::doEndLuminosityBlockProduce_(LuminosityBlock& lbp, EventSetup const& c) {}
    
    void
    EDFilterBase::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;
      desc.setUnknown();
      descriptions.addDefault(desc);
    }
    
    void
    EDFilterBase::prevalidate(ConfigurationDescriptions& iConfig) {
      edmodule_mightGet_config(iConfig);
    }
    
    static const std::string kBaseType("EDFilter");
    
    const std::string&
    EDFilterBase::baseType() {
      return kBaseType;
    }

  }
}
