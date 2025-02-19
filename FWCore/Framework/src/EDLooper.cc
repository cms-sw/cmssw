// -*- C++ -*-
//
// Package:     <package>
// Module:      EDLooper
// 
// Author:      Valentin Kuznetsov
// Created:     Wed Jul  5 11:44:26 EDT 2006

#include "FWCore/Framework/interface/EDLooper.h"
namespace edm {

  EDLooper::EDLooper() : EDLooperBase(){ }
  EDLooper::~EDLooper() { }

  EDLooperBase::Status 
  EDLooper::duringLoop(Event const& iEvent, EventSetup const& iEventSetup, ProcessingController&) {
    return duringLoop(iEvent, iEventSetup);
  }
}
