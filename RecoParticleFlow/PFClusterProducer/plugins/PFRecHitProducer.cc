#include "RecoParticleFlow/PFClusterProducer/plugins/PFRecHitProducer.h"

#include <memory>

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

// For RecHits calibration wrt 50 GeV pions.
// #include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPFCorrsRcd.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
using namespace std;
using namespace edm;


PFRecHitProducer::PFRecHitProducer(const edm::ParameterSet& iConfig)
{

    
  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);

  thresh_Barrel_ = 
    iConfig.getParameter<double>("thresh_Barrel");
  thresh_Endcap_ = 
    iConfig.getParameter<double>("thresh_Endcap");
    
    
  
  //register products
  produces<reco::PFRecHitCollection>();
  produces<reco::PFRecHitCollection>("Cleaned");
  
}


void PFRecHitProducer::produce(edm::Event& iEvent, 
			       const edm::EventSetup& iSetup) {


  auto_ptr< vector<reco::PFRecHit> > recHits( new vector<reco::PFRecHit> ); 
  auto_ptr< vector<reco::PFRecHit> > recHitsCleaned( new vector<reco::PFRecHit> ); 
  
  // fill the collection of rechits (see child classes)
  createRecHits( *recHits, *recHitsCleaned, iEvent, iSetup);

  iEvent.put( recHits );
  iEvent.put( recHitsCleaned, "Cleaned" );

}


PFRecHitProducer::~PFRecHitProducer() {}

// ------------ method called once each job just before starting event loop  ------------
void 
PFRecHitProducer::beginRun(const edm::Run& run,
			   const EventSetup& es) {

  // get the HCAL RecHits correction factors
  // edm::ESHandle<HcalRespCorrs> rchandle;
  // es.get<HcalRespCorrsRcd>().get(rchandle);
  // myRespCorr= rchandle.product();
  // And the PF-specific ones
  edm::ESHandle<HcalPFCorrs> pfrchandle;
  es.get<HcalPFCorrsRcd>().get(pfrchandle);
  myPFCorr= pfrchandle.product();

  // Get cleaned channels in the HCAL and HF 
  // HCAL channel status map ****************************************
  edm::ESHandle<HcalChannelQuality> hcalChStatus;    
  es.get<HcalChannelQualityRcd>().get( hcalChStatus );
  theHcalChStatus = hcalChStatus.product();

  // Retrieve the good/bad ECAL channels from the DB
  edm::ESHandle<EcalChannelStatus> ecalChStatus;
  es.get<EcalChannelStatusRcd>().get(ecalChStatus);
  theEcalChStatus = ecalChStatus.product();

  edm::ESHandle<CaloTowerConstituentsMap> cttopo;
  es.get<IdealGeometryRecord>().get(cttopo);
  theTowerConstituentsMap = cttopo.product();
}


//define this as a plug-in
// DEFINE_FWK_MODULE(PFRecHitProducer);

