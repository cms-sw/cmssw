/**_________________________________________________________________
   class:   AlcaBeamSpotFromDB.cc
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: AlcaBeamSpotFromDB.cc,v 1.4 2013/05/17 20:25:11 chrjones Exp $

________________________________________________________________**/


// C++ standard
#include <string>
// CMS
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "Calibration/TkAlCaRecoProducers/interface/AlcaBeamSpotFromDB.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"


AlcaBeamSpotFromDB::AlcaBeamSpotFromDB(const edm::ParameterSet& iConfig)
{

  produces<reco::BeamSpot, edm::InLumi>("alcaBeamSpot");  
}


AlcaBeamSpotFromDB::~AlcaBeamSpotFromDB()
{
	
}

//--------------------------------------------------------------------------------------------------                                                                      
void AlcaBeamSpotFromDB::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){

}

//--------------------------------------------------------------------------------------------------                                                                      
void AlcaBeamSpotFromDB::endLuminosityBlockProduce(edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup)
{
  // read DB object
  edm::ESHandle< BeamSpotObjects > beamhandle;
  iSetup.get<BeamSpotObjectsRcd>().get(beamhandle);
  const BeamSpotObjects *spotDB = beamhandle.product();

  // translate from BeamSpotObjects to reco::BeamSpot
  reco::BeamSpot::Point apoint( spotDB->GetX(), spotDB->GetY(), spotDB->GetZ() );
  
  reco::BeamSpot::CovarianceMatrix matrix;
  for ( int i=0; i<7; ++i ) {
    for ( int j=0; j<7; ++j ) {
      matrix(i,j) = spotDB->GetCovariance(i,j);
    }
  }
  
  reco::BeamSpot aSpot;
  // this assume beam width same in x and y
  aSpot = reco::BeamSpot( apoint,
			  spotDB->GetSigmaZ(),
			  spotDB->Getdxdz(),
			  spotDB->Getdydz(),
			  spotDB->GetBeamWidthX(),
			  matrix );
  aSpot.setBeamWidthY( spotDB->GetBeamWidthY() );
  aSpot.setEmittanceX( spotDB->GetEmittanceX() );
  aSpot.setEmittanceY( spotDB->GetEmittanceY() );
  aSpot.setbetaStar( spotDB->GetBetaStar() );

  if ( spotDB->GetBeamType() == 2 ) {
    aSpot.setType( reco::BeamSpot::Tracker );
  } else{
    aSpot.setType( reco::BeamSpot::Fake );
  }

  std::auto_ptr<reco::BeamSpot> result(new reco::BeamSpot);
  *result = aSpot;
  lumiSeg.put(result, std::string("alcaBeamSpot"));

  //std::cout << " for runs: " << iEvent.id().run() << " - " << iEvent.id().run() << std::endl;
  std::cout << aSpot << std::endl;

}


void
AlcaBeamSpotFromDB::beginJob()
{
}

void
AlcaBeamSpotFromDB::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(AlcaBeamSpotFromDB);
