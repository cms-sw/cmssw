// -*- C++ -*-
//
// Package:    CheckPhase2Cabling
// Class:      CheckPhase2Cabling
//
/**\class CheckPhase2Cabling CheckPhase2Cabling.cc CalibTracker/CheckPhase2Cabling/src/CheckPhase2Cabling.cc

 Description: simple check of the phase2 cabling

 Implementation:
     Just dump the cabling information
*/
//
// Original Author:  Christophe Delaere
//         Created:  Fri Dec 20 19:37:34 CET 2013
//

// system include files
#include <memory>
#include <iostream>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/SiStripObjects/interface/Phase2TrackerModule.h"
#include "CondFormats/SiStripObjects/interface/Phase2TrackerCabling.h"
#include "CondFormats/DataRecord/interface/Phase2TrackerCablingRcd.h"

//
// class declaration
//

class CheckPhase2Cabling : public edm::one::EDAnalyzer<> {
public:
  explicit CheckPhase2Cabling(const edm::ParameterSet&);
  ~CheckPhase2Cabling();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  // ----------member data ---------------------------
  const edm::ESGetToken<Phase2TrackerCabling, Phase2TrackerCablingRcd> cablingToken_;
};

//
// constructors and destructor
//
CheckPhase2Cabling::CheckPhase2Cabling(const edm::ParameterSet& iConfig) : cablingToken_(esConsumes()) {}

CheckPhase2Cabling::~CheckPhase2Cabling() = default;
//
// member functions
//

// ------------ method called for each event  ------------
void CheckPhase2Cabling::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  ESHandle<Phase2TrackerCabling> cablingHandle = iSetup.getHandle(cablingToken_);

  // print general information about the cabling
  std::cout << cablingHandle->summaryDescription() << std::endl;
  // print detailed information
  std::cout << cablingHandle->description() << std::endl;

  // search information about one module
  Phase2TrackerModule module = cablingHandle->findFedCh(std::make_pair(0, 1));

  // print information about that module
  std::cout << "Information about the module connected to FED 0.1:" << std::endl;
  std::cout << module.description() << std::endl;

  // look at one subset (based on cooling)
  Phase2TrackerCabling coolingLoop = cablingHandle->filterByCoolingLine(0);
  std::cout << "Subset in cooling line 0:" << std::endl;
  std::cout << coolingLoop.description(1) << std::endl;

  // look at one subset (based on power)
  Phase2TrackerCabling powerGroup = cablingHandle->filterByPowerGroup(1);
  std::cout << "Subset in power group 1:" << std::endl;
  std::cout << powerGroup.description(1) << std::endl;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void CheckPhase2Cabling::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(CheckPhase2Cabling);
