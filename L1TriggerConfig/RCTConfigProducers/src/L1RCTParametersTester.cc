// -*- C++ -*-
//
// Package:    RCTConfigTester
// Class:      RCTConfigTester
//
/**\class RCTConfigTester RCTConfigTester.h L1TriggerConfig/RCTConfigTester/src/RCTConfigTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sridhara Dasu
//         Created:  Mon Jul 16 23:48:35 CEST 2007
//
//
// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>

#include <iostream>

using std::cout;
using std::endl;
//
// class declaration
//

class L1RCTParametersTester : public edm::EDAnalyzer {
public:
  explicit L1RCTParametersTester(const edm::ParameterSet&) {}
  ~L1RCTParametersTester() override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;
};

void L1RCTParametersTester::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  edm::ESHandle<L1RCTParameters> rctParam;
  evSetup.get<L1RCTParametersRcd>().get(rctParam);

  rctParam->print(std::cout);
}

DEFINE_FWK_MODULE(L1RCTParametersTester);
