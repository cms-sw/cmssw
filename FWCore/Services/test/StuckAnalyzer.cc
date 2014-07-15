// -*- C++ -*-
//
// Package:     Subsystem/Package
// Class  :     StuckAnalyzer
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Sat, 22 Mar 2014 23:07:05 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class StuckAnalyzer : public edm::global::EDAnalyzer<> {
public:
  StuckAnalyzer(edm::ParameterSet const&) {}
  
  void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override {
    while(true) {};
  }
};


DEFINE_FWK_MODULE(StuckAnalyzer);
