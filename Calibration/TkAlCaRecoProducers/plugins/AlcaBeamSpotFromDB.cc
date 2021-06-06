/**_________________________________________________________________
   class:   AlcaBeamSpotFromDB.cc
   package: RecoVertex/BeamSpotProducer



 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)


________________________________________________________________**/

// C++ standard
#include <string>
// CMS
#include "Calibration/TkAlCaRecoProducers/interface/AlcaBeamSpotFromDB.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

AlcaBeamSpotFromDB::AlcaBeamSpotFromDB(const edm::ParameterSet &iConfig)
    : beamSpotToken_(esConsumes<BeamSpotObjects, BeamSpotObjectsRcd, edm::Transition::EndLuminosityBlock>()) {
  produces<reco::BeamSpot, edm::Transition::EndLuminosityBlock>("alcaBeamSpot");
}

AlcaBeamSpotFromDB::~AlcaBeamSpotFromDB() {}

//--------------------------------------------------------------------------------------------------
void AlcaBeamSpotFromDB::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {}

//--------------------------------------------------------------------------------------------------
void AlcaBeamSpotFromDB::endLuminosityBlockProduce(edm::LuminosityBlock &lumiSeg, const edm::EventSetup &iSetup) {
  // read DB object
  const BeamSpotObjects *spotDB = &iSetup.getData(beamSpotToken_);

  // translate from BeamSpotObjects to reco::BeamSpot
  reco::BeamSpot::Point apoint(spotDB->GetX(), spotDB->GetY(), spotDB->GetZ());

  reco::BeamSpot::CovarianceMatrix matrix;
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 7; ++j) {
      matrix(i, j) = spotDB->GetCovariance(i, j);
    }
  }

  reco::BeamSpot aSpot;
  // this assume beam width same in x and y
  aSpot = reco::BeamSpot(
      apoint, spotDB->GetSigmaZ(), spotDB->Getdxdz(), spotDB->Getdydz(), spotDB->GetBeamWidthX(), matrix);
  aSpot.setBeamWidthY(spotDB->GetBeamWidthY());
  aSpot.setEmittanceX(spotDB->GetEmittanceX());
  aSpot.setEmittanceY(spotDB->GetEmittanceY());
  aSpot.setbetaStar(spotDB->GetBetaStar());

  if (spotDB->GetBeamType() == 2) {
    aSpot.setType(reco::BeamSpot::Tracker);
  } else {
    aSpot.setType(reco::BeamSpot::Fake);
  }

  auto result = std::make_unique<reco::BeamSpot>();
  *result = aSpot;
  lumiSeg.put(std::move(result), std::string("alcaBeamSpot"));

  // std::cout << " for runs: " << iEvent.id().run() << " - " <<
  // iEvent.id().run() << std::endl;
  std::cout << aSpot << std::endl;
}

void AlcaBeamSpotFromDB::beginJob() {}

void AlcaBeamSpotFromDB::endJob() {}

// define this as a plug-in
DEFINE_FWK_MODULE(AlcaBeamSpotFromDB);
