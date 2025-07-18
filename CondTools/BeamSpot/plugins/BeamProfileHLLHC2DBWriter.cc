// -*- C++ -*-
//
// Package:    BeamProfileHLLHC2DBWriter
// Class:      BeamProfileHLLHC2DBWriter
//
/**\class BeamProfileHLLHC2DBWriter BeamProfileHLLHC2DBWriter.cc CondTools/BeamSpot/plugins/BeamProfileHLLHC2DBWriter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Francesco Brivio (INFN Milano-Bicocca)
//         Created:  November 2, 2023
//

// system include files
#include <memory>

// user include files
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/BeamSpotObjects/interface/SimBeamSpotHLLHCObjects.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

//
// class declaration
//
class BeamProfileHLLHC2DBWriter : public edm::global::EDAnalyzer<> {
public:
  explicit BeamProfileHLLHC2DBWriter(const edm::ParameterSet&);
  ~BeamProfileHLLHC2DBWriter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override;
  void endJob() override;

  // ----------member data ---------------------------
  const std::string recordName_;
  SimBeamSpotHLLHCObjects beamSpot_;
};

// ------------ constructor  ------------
BeamProfileHLLHC2DBWriter::BeamProfileHLLHC2DBWriter(const edm::ParameterSet& iConfig)
    : recordName_(iConfig.getParameter<std::string>("recordName")) {
  beamSpot_.setMeanX(iConfig.getParameter<double>("MeanX"));
  beamSpot_.setMeanY(iConfig.getParameter<double>("MeanY"));
  beamSpot_.setMeanZ(iConfig.getParameter<double>("MeanZ"));
  beamSpot_.setEProton(iConfig.getParameter<double>("EProton"));
  beamSpot_.setCrabFrequency(iConfig.getParameter<double>("CrabFrequency"));
  beamSpot_.setRF800(iConfig.getParameter<double>("RF800"));
  beamSpot_.setCrossingAngle(iConfig.getParameter<double>("CrossingAngle"));
  beamSpot_.setCrabbingAngleCrossing(iConfig.getParameter<double>("CrabbingAngleCrossing"));
  beamSpot_.setCrabbingAngleSeparation(iConfig.getParameter<double>("CrabbingAngleSeparation"));
  beamSpot_.setBetaCrossingPlane(iConfig.getParameter<double>("BetaCrossingPlane"));
  beamSpot_.setBetaSeparationPlane(iConfig.getParameter<double>("BetaSeparationPlane"));
  beamSpot_.setHorizontalEmittance(iConfig.getParameter<double>("HorizontalEmittance"));
  beamSpot_.setVerticalEmittance(iConfig.getParameter<double>("VerticalEmittance"));
  beamSpot_.setBunchLength(iConfig.getParameter<double>("BunchLength"));
  beamSpot_.setTimeOffset(iConfig.getParameter<double>("TimeOffset"));
}

// ------------ method called for each event  ------------
void BeamProfileHLLHC2DBWriter::analyze(edm::StreamID, const edm::Event& iEvent, const edm::EventSetup& iSetup) const {}

// ------------ method called once each job just after ending the event loop  ------------
void BeamProfileHLLHC2DBWriter::endJob() {
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  poolDbService->createOneIOV<SimBeamSpotHLLHCObjects>(beamSpot_, poolDbService->beginOfTime(), recordName_);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BeamProfileHLLHC2DBWriter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("recordName", "SimBeamSpotHLLHCObjectsRcd")
      ->setComment("name of the record to use for the PoolDBOutputService");
  desc.add<double>("MeanX", 0.0)->setComment("in cm");
  desc.add<double>("MeanY", 0.0)->setComment("in cm");
  desc.add<double>("MeanZ", 0.0)->setComment("in cm");
  desc.add<double>("EProton", 0.0)->setComment("in GeV");
  desc.add<double>("CrabFrequency", 0.0)->setComment("in MHz");
  desc.add<double>("RF800", 0.0)->setComment("800 MHz RF?");
  desc.add<double>("CrossingAngle", 0.0)->setComment("in urad");
  desc.add<double>("CrabbingAngleCrossing", 0.0)->setComment("in urad");
  desc.add<double>("CrabbingAngleSeparation", 0.0)->setComment("in urad");
  desc.add<double>("BetaCrossingPlane", 0.0)->setComment("in m");
  desc.add<double>("BetaSeparationPlane", 0.0)->setComment("in m");
  desc.add<double>("HorizontalEmittance", 0.0)->setComment("in mm");
  desc.add<double>("VerticalEmittance", 0.0)->setComment("in mm");
  desc.add<double>("BunchLength", 0.0)->setComment("in m");
  desc.add<double>("TimeOffset", 0.0)->setComment("in ns");
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(BeamProfileHLLHC2DBWriter);
