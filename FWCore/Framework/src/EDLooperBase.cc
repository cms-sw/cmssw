// -*- C++ -*-
//
// Package:     <package>
// Module:      EDLooperBase
// 
// Author:      Valentin Kuznetsov
// Created:     Wed Jul  5 11:44:26 EDT 2006
// $Id: EDLooperBase.cc,v 1.2 2010/09/01 18:26:27 chrjones Exp $

#include "FWCore/Framework/interface/EDLooperBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/MessageLogger/interface/ExceptionMessages.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ScheduleInfo.h"

#include "boost/bind.hpp"


namespace edm {

  EDLooperBase::EDLooperBase() : iCounter_(0), act_table_(0), moduleChanger_(0) { }
  EDLooperBase::~EDLooperBase() { }

  void
  EDLooperBase::doStartingNewLoop() {
    startingNewLoop(iCounter_);
  }

  EDLooperBase::Status
  EDLooperBase::doDuringLoop(edm::EventPrincipal& eventPrincipal, const edm::EventSetup& es, edm::ProcessingController& ioController) {
    edm::ModuleDescription modDesc("EDLooperBase", "");
    Event event(eventPrincipal, modDesc);

    Status status = kContinue;
    try {
      status = duringLoop(event, es, ioController);
    }
    catch(cms::Exception& e) {
      e.addContext("Calling the 'duringLoop' method of a looper");
      actions::ActionCodes action = (act_table_->find(e.category()));
      if (action != actions::Rethrow) {
        edm::printCmsExceptionWarning("SkipEvent", e);
      }
      else {
        throw;
      }
    }
    return status;
  }

  EDLooperBase::Status
  EDLooperBase::doEndOfLoop(const edm::EventSetup& es) {
    return endOfLoop(es, iCounter_);
  }

  void
  EDLooperBase::prepareForNextLoop(eventsetup::EventSetupProvider* esp) {
    ++iCounter_;

    const std::set<edm::eventsetup::EventSetupRecordKey>& keys = modifyingRecords();
    for_all(keys,
      boost::bind(&eventsetup::EventSetupProvider::resetRecordPlusDependentRecords,
                  esp, _1));
  }

  void EDLooperBase::beginOfJob(const edm::EventSetup&) { beginOfJob();}
  void EDLooperBase::beginOfJob() { }

  void EDLooperBase::endOfJob() { }

  void EDLooperBase::doBeginRun(RunPrincipal& iRP, EventSetup const& iES){
        edm::ModuleDescription modDesc("EDLooperBase", "");
	Run run(iRP, modDesc);
	beginRun(run,iES);
  }

  void EDLooperBase::doEndRun(RunPrincipal& iRP, EventSetup const& iES){
        edm::ModuleDescription modDesc("EDLooperBase", "");
	Run run(iRP, modDesc);
	endRun(run,iES);
  }
  void EDLooperBase::doBeginLuminosityBlock(LuminosityBlockPrincipal& iLB, EventSetup const& iES){
    edm::ModuleDescription modDesc("EDLooperBase", "");
    LuminosityBlock luminosityBlock(iLB, modDesc);
    beginLuminosityBlock(luminosityBlock,iES);
  }
  void EDLooperBase::doEndLuminosityBlock(LuminosityBlockPrincipal& iLB, EventSetup const& iES){
    edm::ModuleDescription modDesc("EDLooperBase", "");
    LuminosityBlock luminosityBlock(iLB, modDesc);
    endLuminosityBlock(luminosityBlock,iES);
  }

  void EDLooperBase::beginRun(Run const&, EventSetup const&){}
  void EDLooperBase::endRun(Run const&, EventSetup const&){}
  void EDLooperBase::beginLuminosityBlock(LuminosityBlock const&, EventSetup const&){}
  void EDLooperBase::endLuminosityBlock(LuminosityBlock const&, EventSetup const&){}

  void EDLooperBase::attachTo(ActivityRegistry&){}
   

  std::set<eventsetup::EventSetupRecordKey> 
  EDLooperBase::modifyingRecords() const
  {
    return std::set<eventsetup::EventSetupRecordKey> ();
  }
   
  void 
  EDLooperBase::copyInfo(const ScheduleInfo& iInfo){
    scheduleInfo_ = std::auto_ptr<ScheduleInfo>(new ScheduleInfo(iInfo));
  }
  void 
  EDLooperBase::setModuleChanger(const ModuleChanger* iChanger) {
    moduleChanger_ = iChanger;
  }

  const ModuleChanger* EDLooperBase::moduleChanger() const {
    return moduleChanger_;
  }
  const ScheduleInfo* EDLooperBase::scheduleInfo() const {
    return scheduleInfo_.get();
  }
  
}
