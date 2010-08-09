// -*- C++ -*-
//
// Package:     <package>
// Module:      EDLooper
// 
// Author:      Valentin Kuznetsov
// Created:     Wed Jul  5 11:44:26 EDT 2006
// $Id: EDLooper.cc,v 1.15 2010/07/22 15:00:28 chrjones Exp $

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
#include "FWCore/Framework/interface/ScheduleInfo.h"

#include "boost/bind.hpp"


namespace edm {

  EDLooper::EDLooper() : EDLooperBase(){ }
  EDLooper::~EDLooper() { }

  EDLooperBase::Status 
  EDLooper::duringLoop(const edm::Event& iEvent, const edm::EventSetup& iEventSetup, edm::ProcessingController& ) {
    return duringLoop(iEvent, iEventSetup);
  }

  
}
