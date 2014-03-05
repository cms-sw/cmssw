// -*- C++ -*-
//
// Package:    EBRecoSummary
// Class:      EBRecoSummary
// Original Author:  Martina Malberti
// 
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Common/interface/EventBase.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalTools.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalCleaningAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalRecHitLess.h"

#include "DQMOffline/Ecal/interface/EBRecoSummary.h"

#include <iostream>
#include <cmath>
#include <string>

//
// constructors and destructor
//
EBRecoSummary::EBRecoSummary(const edm::ParameterSet& ps)
{

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  //now do what ever initialization is needed
  recHitCollection_EB_       = consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("recHitCollection_EB"));
  redRecHitCollection_EB_    = consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("redRecHitCollection_EB"));
  basicClusterCollection_EB_ = consumes<edm::View<reco::CaloCluster> >(ps.getParameter<edm::InputTag>("basicClusterCollection_EB"));
  superClusterCollection_EB_ = consumes<reco::SuperClusterCollection>(ps.getParameter<edm::InputTag>("superClusterCollection_EB"));

  std::cout << "EBRecoSummary " << ps.getParameter<edm::InputTag>("basicClusterCollection_EB") << std::endl;

  ethrEB_                    = ps.getParameter<double>("ethrEB");

  scEtThrEB_                 = ps.getParameter<double>("scEtThrEB");

  // DQM Store -------------------
  dqmStore_= edm::Service<DQMStore>().operator->();

  // Monitor Elements (ex THXD)
  dqmStore_->setCurrentFolder(prefixME_ + "/EBRecoSummary"); // to organise the histos in folders
     
  // ReducedRecHits ----------------------------------------------
  // ... barrel 
  h_redRecHits_EB_recoFlag = dqmStore_->book1D("redRecHits_EB_recoFlag","redRecHits_EB_recoFlag",16,-0.5,15.5);  

  // RecHits ---------------------------------------------- 
  // ... barrel
  h_recHits_EB_energyMax     = dqmStore_->book1D("recHits_EB_energyMax","recHitsEB_energyMax",110,-10,100);
  h_recHits_EB_Chi2          = dqmStore_->book1D("recHits_EB_Chi2","recHits_EB_Chi2",200,0,100);
  h_recHits_EB_time          = dqmStore_->book1D("recHits_EB_time","recHits_EB_time",200,-50,50);
  h_recHits_EB_E1oE4         = dqmStore_->book1D("recHits_EB_E1oE4","recHitsEB_E1oE4",150, 0, 1.5);
  h_recHits_EB_recoFlag      = dqmStore_->book1D("recHits_EB_recoFlag","recHits_EB_recoFlag",16,-0.5,15.5);  

  // Basic Clusters ----------------------------------------------    
  // ... associated barrel rec hits
  h_basicClusters_recHits_EB_recoFlag = dqmStore_->book1D("basicClusters_recHits_EB_recoFlag","basicClusters_recHits_EB_recoFlag",16,-0.5,15.5);  

  // Super Clusters ----------------------------------------------
  // ... barrel
  h_superClusters_EB_nBC     = dqmStore_->book1D("superClusters_EB_nBC","superClusters_EB_nBC",100,0.,100.);
  h_superClusters_EB_E1oE4   = dqmStore_->book1D("superClusters_EB_E1oE4","superClusters_EB_E1oE4",150,0,1.5);

  h_superClusters_eta        = dqmStore_->book1D("superClusters_eta","superClusters_eta",150,-3.,3.);
  h_superClusters_EB_phi     = dqmStore_->book1D("superClusters_EB_phi","superClusters_EB_phi",360,-3.1415927,3.1415927);
  
}



EBRecoSummary::~EBRecoSummary()
{
        // do anything here that needs to be done at desctruction time
        // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to for each event  ------------
void EBRecoSummary::analyze(const edm::Event& ev, const edm::EventSetup& iSetup)
{
  // calo topology
  edm::ESHandle<CaloTopology> pTopology;
  iSetup.get<CaloTopologyRecord>().get(pTopology);
  const CaloTopology *topology = pTopology.product();

  // --- REDUCED REC HITS ------------------------------------------------------------------------------------- 
  edm::Handle<EcalRecHitCollection> redRecHitsEB;
  ev.getByToken( redRecHitCollection_EB_, redRecHitsEB );
  const EcalRecHitCollection* theBarrelEcalredRecHits = redRecHitsEB.product () ;
  if ( ! redRecHitsEB.isValid() ) {
    edm::LogWarning("EBRecoSummary") << "redRecHitsEB not found"; 
  }
  
  for ( EcalRecHitCollection::const_iterator itr = theBarrelEcalredRecHits->begin () ;
        itr != theBarrelEcalredRecHits->end () ;++itr)
  {
      
    h_redRecHits_EB_recoFlag->Fill( itr -> recoFlag() );
  
  }

  // --- REC HITS ------------------------------------------------------------------------------------- 
  
  // ... barrel
  edm::Handle<EcalRecHitCollection> recHitsEB;
  ev.getByToken( recHitCollection_EB_, recHitsEB );
  const EcalRecHitCollection* theBarrelEcalRecHits = recHitsEB.product () ;
  if ( ! recHitsEB.isValid() ) {
    edm::LogWarning("EBRecoSummary") << "recHitsEB not found"; 
  }

  float maxRecHitEnergyEB = -999.;
  
  EBDetId ebid_MrecHitEB;

  for ( EcalRecHitCollection::const_iterator itr = theBarrelEcalRecHits->begin () ;
	itr != theBarrelEcalRecHits->end () ;++itr)
    {

      EBDetId ebid( itr -> id() );
      
      h_recHits_EB_recoFlag      -> Fill( itr -> recoFlag() );

      // max E rec hit
      if (itr -> energy() > maxRecHitEnergyEB ){
	maxRecHitEnergyEB = itr -> energy() ;
      }       

      if ( itr -> energy() > ethrEB_ ){
	h_recHits_EB_Chi2          -> Fill( itr -> chi2() );
        h_recHits_EB_time          -> Fill( itr -> time() );
      }

      float R4 = EcalTools::swissCross( ebid, *theBarrelEcalRecHits, 0. );
      
      if ( itr -> energy() > 3. && abs(ebid.ieta())!=85 )  h_recHits_EB_E1oE4-> Fill( R4 );
      
    }
  
  h_recHits_EB_energyMax         -> Fill( maxRecHitEnergyEB );
  
  //--- BASIC CLUSTERS --------------------------------------------------------------

  // ... barrel
  edm::Handle<edm::View<reco::CaloCluster> > basicClusters_EB_h;
  if(ev.getByToken( basicClusterCollection_EB_, basicClusters_EB_h )){

    const edm::View<reco::CaloCluster>* theBarrelBasicClusters = basicClusters_EB_h.product () ;

    for (edm::View<reco::CaloCluster>::const_iterator itBC = theBarrelBasicClusters->begin(); 
         itBC != theBarrelBasicClusters->end(); ++itBC ) {
         
      //Get the associated RecHits
      const std::vector<std::pair<DetId,float> > & hits= itBC->hitsAndFractions();
      for (std::vector<std::pair<DetId,float> > ::const_iterator rh = hits.begin(); rh!=hits.end(); ++rh){
      
        EBRecHitCollection::const_iterator itrechit = theBarrelEcalRecHits->find((*rh).first);
        if (itrechit==theBarrelEcalRecHits->end()) continue;
        h_basicClusters_recHits_EB_recoFlag -> Fill ( itrechit -> recoFlag() );
    
      }
  
    }
  }
  else{
    //    edm::LogWarning("EBRecoSummary") << "basicClusters_EB_h not found"; 
  }
 
  // Super Clusters
  // ... barrel
  edm::Handle<reco::SuperClusterCollection> superClusters_EB_h;
  ev.getByToken( superClusterCollection_EB_, superClusters_EB_h );
  const reco::SuperClusterCollection* theBarrelSuperClusters = superClusters_EB_h.product () ;
  if ( ! superClusters_EB_h.isValid() ) {
    edm::LogWarning("EBRecoSummary") << "superClusters_EB_h not found"; 
  }

  for (reco::SuperClusterCollection::const_iterator itSC = theBarrelSuperClusters->begin(); 
       itSC != theBarrelSuperClusters->end(); ++itSC ) {
    
    double scEt = itSC -> energy() * sin(2.*atan( exp(- itSC->position().eta() )));
    
    if (scEt < scEtThrEB_ ) continue;

    h_superClusters_EB_nBC    -> Fill( itSC -> clustersSize());
    h_superClusters_eta       -> Fill( itSC -> eta() );
    h_superClusters_EB_phi    -> Fill( itSC -> phi() );
 
    float E1 = EcalClusterTools::eMax   ( *itSC, theBarrelEcalRecHits);
    float E4 = EcalClusterTools::eTop   ( *itSC, theBarrelEcalRecHits, topology )+
               EcalClusterTools::eRight ( *itSC, theBarrelEcalRecHits, topology )+
               EcalClusterTools::eBottom( *itSC, theBarrelEcalRecHits, topology )+
               EcalClusterTools::eLeft  ( *itSC, theBarrelEcalRecHits, topology );

    if ( E1 > 3. ) h_superClusters_EB_E1oE4  -> Fill( 1.- E4/E1);
    
  }

}


// ------------ method called once each job just before starting event loop  ------------
        void 
EBRecoSummary::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EBRecoSummary::endJob() 
{}

// ----------additional functions-------------------

void EBRecoSummary::convxtalid(Int_t &nphi,Int_t &neta)
{
  // Barrel only
  // Output nphi 0...359; neta 0...84; nside=+1 (for eta>0), or 0 (for eta<0).
  // neta will be [-85,-1] , or [0,84], the minus sign indicates the z<0 side.
  
  if(neta > 0) neta -= 1;
  if(nphi > 359) nphi=nphi-360;
  
} //end of convxtalid

int EBRecoSummary::diff_neta_s(Int_t neta1, Int_t neta2){
  Int_t mdiff;
  mdiff=(neta1-neta2);
  return mdiff;
}

// Calculate the distance in xtals taking into account the periodicity of the Barrel
int EBRecoSummary::diff_nphi_s(Int_t nphi1,Int_t nphi2) {
  Int_t mdiff;
  if(abs(nphi1-nphi2) < (360-abs(nphi1-nphi2))) {
    mdiff=nphi1-nphi2;
  }
  else {
    mdiff=360-abs(nphi1-nphi2);
    if(nphi1>nphi2) mdiff=-mdiff;
  }
  return mdiff;
}

//define this as a plug-in
DEFINE_FWK_MODULE(EBRecoSummary);
