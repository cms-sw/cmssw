// -*- C++ -*-
//
// Package:    BeamProfile2DBWriter
// Class:      BeamProfile2DBWriter
//
/**\class BeamProfile2DBWriter BeamProfile2DBWriter.cc CondTools/BeamSpot/plugins/BeamProfile2DBWriter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jean-Roch Vlimant,40 3-A28,+41227671209,
//         Created:  Fri Jan  6 14:49:42 CET 2012
//
// Updated; Francesco Brivio, June 11, 2023
//

// system include files
#include <memory>

// user include files
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/BeamSpotObjects/interface/SimBeamSpotObjects.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <CLHEP/Units/SystemOfUnits.h>
#include <CLHEP/Units/GlobalPhysicalConstants.h>

//
// class declaration
//
class BeamProfile2DBWriter : public edm::global::EDAnalyzer<> {
public:
  explicit BeamProfile2DBWriter(const edm::ParameterSet&);
  ~BeamProfile2DBWriter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override;
  void endJob() override;

  // ----------member data ---------------------------
  const std::string recordName_;
  SimBeamSpotObjects beamSpot_;
};

//
// constructors and destructor
//
BeamProfile2DBWriter::BeamProfile2DBWriter(const edm::ParameterSet& iConfig)
    : recordName_(iConfig.getParameter<std::string>("recordName")) {
  beamSpot_.setX(iConfig.getParameter<double>("X0"));
  beamSpot_.setY(iConfig.getParameter<double>("Y0"));
  beamSpot_.setZ(iConfig.getParameter<double>("Z0"));
  beamSpot_.setMeanX(iConfig.getParameter<double>("MeanX"));
  beamSpot_.setMeanY(iConfig.getParameter<double>("MeanY"));
  beamSpot_.setMeanZ(iConfig.getParameter<double>("MeanZ"));
  beamSpot_.setSigmaX(iConfig.getParameter<double>("SigmaX"));
  beamSpot_.setSigmaY(iConfig.getParameter<double>("SigmaY"));
  beamSpot_.setSigmaZ(iConfig.getParameter<double>("SigmaZ"));
  beamSpot_.setAlpha(iConfig.getParameter<double>("Alpha"));
  beamSpot_.setPhi(iConfig.getParameter<double>("Phi"));
  beamSpot_.setBetaStar(iConfig.getParameter<double>("BetaStar"));
  beamSpot_.setEmittance(iConfig.getParameter<double>("Emittance"));
  beamSpot_.setTimeOffset(iConfig.getParameter<double>("TimeOffset"));
}

//
// member functions
//

// ------------ method called for each event  ------------
void BeamProfile2DBWriter::analyze(edm::StreamID, const edm::Event& iEvent, const edm::EventSetup& iSetup) const {}

// ------------ method called once each job just after ending the event loop  ------------
void BeamProfile2DBWriter::endJob() {
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  poolDbService->createOneIOV<SimBeamSpotObjects>(beamSpot_, poolDbService->beginOfTime(), recordName_);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BeamProfile2DBWriter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<std::string>("recordName", "SimBeamSpotObjectsRcd")
      ->setComment("name of the record to use for the PoolDBOutputService");
  desc.add<double>("X0", 0.0)->setComment("in cm");
  desc.add<double>("Y0", 0.0)->setComment("in cm");
  desc.add<double>("Z0", 0.0)->setComment("in cm");
  desc.add<double>("MeanX", 0.0)->setComment("in cm");
  desc.add<double>("MeanY", 0.0)->setComment("in cm");
  desc.add<double>("MeanZ", 0.0)->setComment("in cm");
  desc.add<double>("SigmaX", -1.0)->setComment("in cm");
  desc.add<double>("SigmaY", -1.0)->setComment("in cm");
  desc.add<double>("SigmaZ", 0.0)->setComment("in cm");
  desc.add<double>("BetaStar", 0.0)->setComment("in cm");
  desc.add<double>("Emittance", 0.0)->setComment("in cm");
  desc.add<double>("Alpha", 0.0)->setComment("in radians");
  desc.add<double>("Phi", 0.0)->setComment("in radians");
  desc.add<double>("TimeOffset", 0.0)->setComment("in ns");
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(BeamProfile2DBWriter);
