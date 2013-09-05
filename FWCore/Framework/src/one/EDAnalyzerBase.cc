// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     one::EDAnalyzerBase
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Thu, 02 May 2013 21:56:04 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/one/EDAnalyzerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/src/edmodule_mightGet_config.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"


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
    EDAnalyzerBase::EDAnalyzerBase():
    moduleDescription_() { }

    EDAnalyzerBase::~EDAnalyzerBase()
    {
    }
    
    bool
    EDAnalyzerBase::doEvent(EventPrincipal& ep, EventSetup const& c,
                            ModuleCallingContext const* mcc) {
      Event e(ep, moduleDescription_, mcc);
      e.setConsumer(this);
      this->analyze(e, c);
      return true;
    }
    
    void
    EDAnalyzerBase::doBeginJob() {
      this->beginJob();
    }
    
    void
    EDAnalyzerBase::doEndJob() {
      this->endJob();
    }
    
    void
    EDAnalyzerBase::doBeginRun(RunPrincipal& rp, EventSetup const& c,
                               ModuleCallingContext const* mcc) {
      Run r(rp, moduleDescription_, mcc);
      r.setConsumer(this);
      Run const& cnstR = r;
      this->doBeginRun_(cnstR, c);
    }
    
    void
    EDAnalyzerBase::doEndRun(RunPrincipal& rp, EventSetup const& c,
                             ModuleCallingContext const* mcc) {
      Run r(rp, moduleDescription_, mcc);
      r.setConsumer(this);
      Run const& cnstR = r;
      this->doEndRun_(cnstR, c);
    }
    
    void
    EDAnalyzerBase::doBeginLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                                           ModuleCallingContext const* mcc) {
      LuminosityBlock lb(lbp, moduleDescription_, mcc);
      lb.setConsumer(this);
      LuminosityBlock const& cnstLb = lb;
      this->doBeginLuminosityBlock_(cnstLb, c);
    }
    
    void
    EDAnalyzerBase::doEndLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                                         ModuleCallingContext const* mcc) {
      LuminosityBlock lb(lbp, moduleDescription_, mcc);
      lb.setConsumer(this);
      LuminosityBlock const& cnstLb = lb;
      this->doEndLuminosityBlock_(cnstLb, c);
    }
    
    void
    EDAnalyzerBase::doRespondToOpenInputFile(FileBlock const& fb) {
      //respondToOpenInputFile(fb);
    }
    
    void
    EDAnalyzerBase::doRespondToCloseInputFile(FileBlock const& fb) {
      //respondToCloseInputFile(fb);
    }
    
    void
    EDAnalyzerBase::doPreForkReleaseResources() {
      //preForkReleaseResources();
    }
    
    void
    EDAnalyzerBase::doPostForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren) {
      //postForkReacquireResources(iChildIndex, iNumberOfChildren);
    }
    
    void EDAnalyzerBase::doBeginRun_(Run const& rp, EventSetup const& c) {}
    void EDAnalyzerBase::doEndRun_(Run const& rp, EventSetup const& c) {}
    void EDAnalyzerBase::doBeginLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDAnalyzerBase::doEndLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c) {}

    void
    EDAnalyzerBase::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;
      desc.setUnknown();
      descriptions.addDefault(desc);
    }
    
    void
    EDAnalyzerBase::prevalidate(ConfigurationDescriptions& iConfig) {
      edmodule_mightGet_config(iConfig);
    }
    
    static const std::string kBaseType("EDAnalyzer");
    
    const std::string&
    EDAnalyzerBase::baseType() {
      return kBaseType;
    }

    void
    EDAnalyzerBase::registerProductsAndCallbacks(EDAnalyzerBase const*, ProductRegistry* reg) {
      
      if (callWhenNewProductsRegistered_) {
        
        reg->callForEachBranch(callWhenNewProductsRegistered_);
        
        Service<ConstProductRegistry> regService;
        regService->watchProductAdditions(callWhenNewProductsRegistered_);
      }
    }

  }
}
