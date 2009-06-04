// -*- C++ -*-
//
// Package:     <package>
// Module:      EDLooper
// 
// Author:      Valentin Kuznetsov
// Created:     Wed Jul  5 11:44:26 EDT 2006
// $Id: EDLooper.cc,v 1.13 2009/02/23 20:12:24 chrjones Exp $

#include "FWCore/Framework/interface/EDLooper.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "boost/bind.hpp"


namespace edm {

  EDLooper::EDLooper() : iCounter_(0) { }
  EDLooper::~EDLooper() { }

  void
  EDLooper::doStartingNewLoop() {
    startingNewLoop(iCounter_);
  }

  EDLooper::Status
  EDLooper::doDuringLoop(edm::EventPrincipal& eventPrincipal, const edm::EventSetup& es) {
    edm::ModuleDescription modDesc("EDLooper", "");
    Event event(eventPrincipal, modDesc);

    Status status = kContinue;
    try {
      status = duringLoop(event, es);
    }
    catch(cms::Exception& e) {
      actions::ActionCodes action = (act_table_->find(e.rootCause()));
      if (action != actions::Rethrow) {
        LogWarning(e.category())
          << "An exception occurred in the looper, continuing with the next event\n"
          << e.what();
      }
      else {
        throw;
      }
    }
    return status;
  }

  EDLooper::Status
  EDLooper::doEndOfLoop(const edm::EventSetup& es) {
    return endOfLoop(es, iCounter_);
  }

  void
  EDLooper::prepareForNextLoop(eventsetup::EventSetupProvider* esp) {
    ++iCounter_;

    const std::set<edm::eventsetup::EventSetupRecordKey>& keys = modifyingRecords();
    for_all(keys,
      boost::bind(&eventsetup::EventSetupProvider::resetRecordPlusDependentRecords,
                  esp, _1));
  }

  void EDLooper::beginOfJob(const edm::EventSetup&) { beginOfJob();}
  void EDLooper::beginOfJob() { }

  void EDLooper::endOfJob() { }

  void EDLooper::doBeginRun(RunPrincipal& iRP, EventSetup const& iES){
        edm::ModuleDescription modDesc("EDLooper", "");
	Run run(iRP, modDesc);
	beginRun(run,iES);
  }

  void EDLooper::doEndRun(RunPrincipal& iRP, EventSetup const& iES){
        edm::ModuleDescription modDesc("EDLooper", "");
	Run run(iRP, modDesc);
	endRun(run,iES);
  }
  void EDLooper::doBeginLuminosityBlock(LuminosityBlockPrincipal& iLB, EventSetup const& iES){
    edm::ModuleDescription modDesc("EDLooper", "");
    LuminosityBlock luminosityBlock(iLB, modDesc);
    beginLuminosityBlock(luminosityBlock,iES);
  }
  void EDLooper::doEndLuminosityBlock(LuminosityBlockPrincipal& iLB, EventSetup const& iES){
    edm::ModuleDescription modDesc("EDLooper", "");
    LuminosityBlock luminosityBlock(iLB, modDesc);
    endLuminosityBlock(luminosityBlock,iES);
  }

  void EDLooper::beginRun(Run const&, EventSetup const&){}
  void EDLooper::endRun(Run const&, EventSetup const&){}
  void EDLooper::beginLuminosityBlock(LuminosityBlock const&, EventSetup const&){}
  void EDLooper::endLuminosityBlock(LuminosityBlock const&, EventSetup const&){}


  std::set<eventsetup::EventSetupRecordKey> 
  EDLooper::modifyingRecords() const
  {
    return std::set<eventsetup::EventSetupRecordKey> ();
  }
}
