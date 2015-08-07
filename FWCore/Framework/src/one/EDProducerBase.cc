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
#include "FWCore/Framework/src/edmodule_mightGet_config.h"
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
    EDProducerBase::EDProducerBase():
    ProducerBase(),
    moduleDescription_(),
    previousParentage_(),
    previousParentageId_() { }
    
    EDProducerBase::~EDProducerBase()
    {
    }
    
    bool
    EDProducerBase::doEvent(EventPrincipal& ep, EventSetup const& c,
                            ActivityRegistry* act,
                            ModuleCallingContext const* mcc) {
      Event e(ep, moduleDescription_, mcc);
      e.setConsumer(this);
      {
        //Temporary: switch order of locks to avoid a deadlock with unscheduled
        // proper fix is to releaes resourcesAcquirer when doing a 'getBy*'
        std::lock_guard<SharedResourcesAcquirer> guard(resourcesAcquirer_);
        {
          std::lock_guard<std::mutex> guard(mutex_);
          EventSignalsSentry sentry(act,mcc);
          this->produce(e, c);
        }
        commit_(e,&previousParentage_, &previousParentageId_);
      }
      return true;
    }
    
    SharedResourcesAcquirer EDProducerBase::createAcquirer() {
      return SharedResourcesAcquirer{};
    }

    void
    EDProducerBase::doBeginJob() {
      resourcesAcquirer_ = createAcquirer();
      
      this->beginJob();
    }
    
    void
    EDProducerBase::doEndJob() {
      this->endJob();
    }
    
    void
    EDProducerBase::doBeginRun(RunPrincipal& rp, EventSetup const& c,
                               ModuleCallingContext const* mcc) {
      Run r(rp, moduleDescription_, mcc);
      r.setConsumer(this);
      Run const& cnstR = r;
      this->doBeginRun_(cnstR, c);
      this->doBeginRunProduce_(r,c);
      commit_(r);
    }
    
    void
    EDProducerBase::doEndRun(RunPrincipal& rp, EventSetup const& c,
                             ModuleCallingContext const* mcc) {
      Run r(rp, moduleDescription_, mcc);
      r.setConsumer(this);
      Run const& cnstR = r;
      this->doEndRun_(cnstR, c);
      this->doEndRunProduce_(r, c);
      commit_(r);
    }
    
    void
    EDProducerBase::doBeginLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                                           ModuleCallingContext const* mcc) {
      LuminosityBlock lb(lbp, moduleDescription_, mcc);
      lb.setConsumer(this);
      LuminosityBlock const& cnstLb = lb;
      this->doBeginLuminosityBlock_(cnstLb, c);
      this->doBeginLuminosityBlockProduce_(lb, c);
      commit_(lb);
    }
    
    void
    EDProducerBase::doEndLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                                         ModuleCallingContext const* mcc) {
      LuminosityBlock lb(lbp, moduleDescription_, mcc);
      lb.setConsumer(this);
      LuminosityBlock const& cnstLb = lb;
      this->doEndLuminosityBlock_(cnstLb, c);
      this->doEndLuminosityBlockProduce_(lb, c);
      commit_(lb);
    }
    
    void
    EDProducerBase::doRespondToOpenInputFile(FileBlock const& fb) {
      //respondToOpenInputFile(fb);
    }
    
    void
    EDProducerBase::doRespondToCloseInputFile(FileBlock const& fb) {
      //respondToCloseInputFile(fb);
    }
    
    void
    EDProducerBase::doPreForkReleaseResources() {
      preForkReleaseResources();
    }
    
    void
    EDProducerBase::doPostForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren) {
      postForkReacquireResources(iChildIndex, iNumberOfChildren);
    }
    
    void EDProducerBase::doBeginRun_(Run const& rp, EventSetup const& c) {}
    void EDProducerBase::doEndRun_(Run const& rp, EventSetup const& c) {}
    void EDProducerBase::doBeginLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c) {}
    void EDProducerBase::doEndLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c) {}

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
