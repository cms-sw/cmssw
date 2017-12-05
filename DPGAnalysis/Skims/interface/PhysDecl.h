// -*- C++ -*-
//
// Package:   PhysDecl
// Class:     PhysDecl
//
// Original Author:  Luca Malgeri

#ifndef PhysDecl_H
#define PhysDecl_H

// system include files
#include <memory>
#include <vector>
#include <map>
#include <set>

// user include files
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

//
// class declaration
//


class PhysDecl : public edm::EDFilter {
public:
  explicit PhysDecl( const edm::ParameterSet & );
  ~PhysDecl() override;
  
private:
  bool filter ( edm::Event &, const edm::EventSetup&) override;
  
  bool applyfilter;
  bool debugOn;
  bool init_;
  std::vector<std::string>  hlNames_;  // name of each HLT algorithm
  edm::EDGetTokenT<edm::TriggerResults> hlTriggerResults_;
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> gtDigis_;
};

#endif
