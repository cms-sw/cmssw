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


//
// class declaration
//


class PhysDecl : public edm::EDFilter {
public:
  explicit PhysDecl( const edm::ParameterSet & );
  ~PhysDecl();
  
private:
  virtual bool filter ( edm::Event &, const edm::EventSetup&) override;
  
  bool applyfilter;
  bool debugOn;
  bool init_;
  std::vector<std::string>  hlNames_;  // name of each HLT algorithm
  edm::InputTag hlTriggerResults_;
};

#endif
