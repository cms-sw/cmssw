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
// $Id: EDFilterBase.cc,v 1.1 2013/05/17 14:49:45 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/Framework/interface/one/EDFilterBase.h"
#include "FWCore/Framework/src/CPCSentry.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/src/edmodule_mightGet_config.h"

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
    EDFilterBase::EDFilterBase():
    ProducerBase(),
    moduleDescription_(),
    current_context_(nullptr),
    previousParentage_(),
    previousParentageId_() { }
    
    EDFilterBase::~EDFilterBase()
    {
    }
    
    bool
    EDFilterBase::doEvent(EventPrincipal& ep, EventSetup const& c,
                        CurrentProcessingContext const* cpc) {
      detail::CPCSentry sentry(current_context_, cpc);
      Event e(ep, moduleDescription_);
      e.setConsumer(this);
      bool returnValue = this->filter(e, c);
      commit_(e,&previousParentage_, &previousParentageId_);
      return returnValue;
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
    EDFilterBase::doBeginRun(RunPrincipal& rp, EventSetup const& c,
                           CurrentProcessingContext const* cpc) {
      
      detail::CPCSentry sentry(current_context_, cpc);
      Run r(rp, moduleDescription_);
      r.setConsumer(this);
      Run const& cnstR = r;
      this->doBeginRun_(cnstR, c);
      this->doBeginRunProduce_(r,c);
      commit_(r);
    }
    
    void
    EDFilterBase::doEndRun(RunPrincipal& rp, EventSetup const& c,
                         CurrentProcessingContext const* cpc) {
      detail::CPCSentry sentry(current_context_, cpc);
      Run r(rp, moduleDescription_);
      r.setConsumer(this);
      Run const& cnstR = r;
      this->doEndRun_(cnstR, c);
      this->doEndRunProduce_(r, c);
      commit_(r);
    }
    
    void
    EDFilterBase::doBeginLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                                       CurrentProcessingContext const* cpc) {
      detail::CPCSentry sentry(current_context_, cpc);
      LuminosityBlock lb(lbp, moduleDescription_);
      lb.setConsumer(this);
      LuminosityBlock const& cnstLb = lb;
      this->doBeginLuminosityBlock_(cnstLb, c);
      this->doBeginLuminosityBlockProduce_(lb, c);
      commit_(lb);
    }
    
    void
    EDFilterBase::doEndLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                                     CurrentProcessingContext const* cpc) {
      detail::CPCSentry sentry(current_context_, cpc);
      LuminosityBlock lb(lbp, moduleDescription_);
      lb.setConsumer(this);
      LuminosityBlock const& cnstLb = lb;
      this->doEndLuminosityBlock_(cnstLb, c);
      this->doEndLuminosityBlockProduce_(lb, c);
      commit_(lb);
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
    EDFilterBase::doRespondToOpenOutputFiles(FileBlock const& fb) {
      //respondToOpenOutputFiles(fb);
    }
    
    void
    EDFilterBase::doRespondToCloseOutputFiles(FileBlock const& fb) {
      //respondToCloseOutputFiles(fb);
    }
    
    void
    EDFilterBase::doPreForkReleaseResources() {
      //preForkReleaseResources();
    }
    
    void
    EDFilterBase::doPostForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren) {
      //postForkReacquireResources(iChildIndex, iNumberOfChildren);
    }
    
    void EDFilterBase::doBeginRun_(Run const& rp, EventSetup const& c) {}
    void EDFilterBase::doEndRun_(Run const& rp, EventSetup const& c) {}
    void EDFilterBase::doBeginLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDFilterBase::doEndLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c) {}

    void EDFilterBase::doBeginRunProduce_(Run& rp, EventSetup const& c) {}
    void EDFilterBase::doEndRunProduce_(Run& rp, EventSetup const& c) {}
    void EDFilterBase::doBeginLuminosityBlockProduce_(LuminosityBlock& lbp, EventSetup const& c) {}
    void EDFilterBase::doEndLuminosityBlockProduce_(LuminosityBlock& lbp, EventSetup const& c) {}
    
    CurrentProcessingContext const*
    EDFilterBase::currentContext() const {
      return current_context_;
    }
    
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