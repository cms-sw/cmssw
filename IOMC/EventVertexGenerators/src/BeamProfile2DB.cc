// -*- C++ -*-
//
// Package:    BeamProfile2DB
// Class:      BeamProfile2DB
//
/**\class BeamProfile2DB BeamProfile2DB.cc IOMC/BeamProfile2DB/src/BeamProfile2DB.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jean-Roch Vlimant,40 3-A28,+41227671209,
//         Created:  Fri Jan  6 14:49:42 CET 2012
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/BeamSpotObjects/interface/SimBeamSpotObjects.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

//
// class declaration
//

class BeamProfile2DB : public edm::global::EDAnalyzer<> {
public:
  explicit BeamProfile2DB(const edm::ParameterSet&);
  ~BeamProfile2DB() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override;
  void endJob() override;

  // ----------member data ---------------------------
  SimBeamSpotObjects beamSpot_;
};

namespace {
  SimBeamSpotObjects read(const edm::ParameterSet& p) {
    SimBeamSpotObjects ret;
    ret.fX0 = p.getParameter<double>("X0") * cm;
    ret.fY0 = p.getParameter<double>("Y0") * cm;
    ret.fZ0 = p.getParameter<double>("Z0") * cm;
    ret.fSigmaZ = p.getParameter<double>("SigmaZ") * cm;
    ret.fAlpha = p.getParameter<double>("Alpha") * radian;
    ret.fPhi = p.getParameter<double>("Phi") * radian;
    ret.fbetastar = p.getParameter<double>("BetaStar") * cm;
    ret.femittance = p.getParameter<double>("Emittance") * cm;              // this is not the normalized emittance
    ret.fTimeOffset = p.getParameter<double>("TimeOffset") * ns * c_light;  // HepMC time units are mm
    return ret;
  }

}  // namespace
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
BeamProfile2DB::BeamProfile2DB(const edm::ParameterSet& iConfig) : beamSpot_(read(iConfig)) {}

BeamProfile2DB::~BeamProfile2DB() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called for each event  ------------
void BeamProfile2DB::analyze(edm::StreamID, const edm::Event& iEvent, const edm::EventSetup& iSetup) const {}

// ------------ method called once each job just before starting event loop  ------------
void BeamProfile2DB::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void BeamProfile2DB::endJob() {
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  poolDbService->createNewIOV<SimBeamSpotObjects>(
      &beamSpot_, poolDbService->beginOfTime(), poolDbService->endOfTime(), "SimBeamSpotObjectsRcd");
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BeamProfile2DB::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<double>("X0")->setComment("in cm");
  desc.add<double>("Y0")->setComment("in cm");
  desc.add<double>("Z0")->setComment("in cm");
  desc.add<double>("SigmaZ")->setComment("in cm");
  desc.add<double>("BetaStar")->setComment("in cm");
  desc.add<double>("Emittance")->setComment("in cm");
  desc.add<double>("Alpha")->setComment("in radians");
  desc.add<double>("Phi")->setComment("in radians");
  desc.add<double>("TimeOffset")->setComment("in ns");
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(BeamProfile2DB);
