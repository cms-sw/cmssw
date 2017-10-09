// -*- C++ -*-
//
// Package:   PhysDecl
// Class:     PhysDecl
//
// Original Author:  Luca Malgeri

#ifndef HLTINSPECT_H
#define HLTINSPECT_H

// system include files
#include <memory>
#include <vector>
#include <map>
#include <set>

// user include files
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


//
// class declaration
//


class HLTInspect : public edm::EDAnalyzer {
public:
  explicit HLTInspect( const edm::ParameterSet & );
  ~HLTInspect();
  
private:
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c) override;   
  std::vector<std::string>  hlNames_;  // name of each HLT algorithm
  edm::InputTag hlTriggerResults_;
  bool init_;
};

#endif
