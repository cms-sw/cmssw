// -*- C++ -*-
//
// Package:    EERecoSummary
// Class:      EERecoSummary
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
#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "DataFormats/EgammaReco/interface/PreshowerClusterFwd.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalTools.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalCleaningAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalRecHitLess.h"

#include "DQMOffline/Ecal/interface/EERecoSummary.h"

#include <iostream>
#include <cmath>

//
// constructors and destructor
//
EERecoSummary::EERecoSummary(const edm::ParameterSet& ps)
{

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  //now do what ever initialization is needed
  recHitCollection_EE_       = consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("recHitCollection_EE"));
  redRecHitCollection_EE_    = consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("redRecHitCollection_EE"));
  basicClusterCollection_EE_ = consumes<edm::View<reco::CaloCluster> >(ps.getParameter<edm::InputTag>("basicClusterCollection_EE"));
  superClusterCollection_EE_ = consumes<reco::SuperClusterCollection>(ps.getParameter<edm::InputTag>("superClusterCollection_EE"));

  ethrEE_                    = ps.getParameter<double>("ethrEE");

  scEtThrEE_                 = ps.getParameter<double>("scEtThrEE");

  // DQM Store -------------------
  dqmStore_ = edm::Service<DQMStore>().operator->();

  // Monitor Elements (ex THXD)
  dqmStore_->setCurrentFolder(prefixME_ + "/EERecoSummary"); // to organise the histos in folders
     
  // ReducedRecHits ----------------------------------------------
  // ... endcap 
  h_redRecHits_EE_recoFlag = dqmStore_->book1D("redRecHits_EE_recoFlag","redRecHits_EE_recoFlag",16,-0.5,15.5);  

  // RecHits ---------------------------------------------- 
  // ... endcap
  h_recHits_EE_recoFlag = dqmStore_->book1D("recHits_EE_recoFlag","recHits_EE_recoFlag",16,-0.5,15.5);  

  // ... endcap +
  h_recHits_EEP_energyMax     = dqmStore_->book1D("recHits_EEP_energyMax","recHitsEEP_energyMax",110,-10,100);
  h_recHits_EEP_Chi2          = dqmStore_->book1D("recHits_EEP_Chi2","recHits_EEP_Chi2",200,0,100);
  h_recHits_EEP_time          = dqmStore_->book1D("recHits_EEP_time","recHits_EEP_time",200,-50,50);

  // ... endcap -
  h_recHits_EEM_energyMax     = dqmStore_->book1D("recHits_EEM_energyMax","recHits_EEM_energyMax",110,-10,100);
  h_recHits_EEM_Chi2          = dqmStore_->book1D("recHits_EEM_Chi2","recHits_EEM_Chi2",200,0,100);
  h_recHits_EEM_time          = dqmStore_->book1D("recHits_EEM_time","recHits_EEM_time",200,-50,50);

  // Basic Clusters ----------------------------------------------    
  // ... associated endcap rec hits
  h_basicClusters_recHits_EE_recoFlag = dqmStore_->book1D("basicClusters_recHits_EE_recoFlag","basicClusters_recHits_EE_recoFlag",16,-0.5,15.5);  

  // Super Clusters ----------------------------------------------
  // ... endcap
  h_superClusters_EEP_nBC    = dqmStore_->book1D("superClusters_EEP_nBC","superClusters_EEP_nBC",100,0.,100.);
  h_superClusters_EEM_nBC    = dqmStore_->book1D("superClusters_EEM_nBC","superClusters_EEM_nBC",100,0.,100.);

  h_superClusters_eta        = dqmStore_->book1D("superClusters_eta","superClusters_eta",150,-3.,3.);
  h_superClusters_EE_phi     = dqmStore_->book1D("superClusters_EE_phi","superClusters_EE_phi",360,-3.1415927,3.1415927);
  
}



EERecoSummary::~EERecoSummary()
{
        // do anything here that needs to be done at desctruction time
        // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to for each event  ------------
void EERecoSummary::analyze(const edm::Event& ev, const edm::EventSetup& iSetup)
{
  // --- REDUCED REC HITS ------------------------------------------------------------------------------------- 
  // ... endcap
  edm::Handle<EcalRecHitCollection> redRecHitsEE;
  ev.getByToken( redRecHitCollection_EE_, redRecHitsEE );
  const EcalRecHitCollection* theEndcapEcalredRecHits = redRecHitsEE.product () ;
  if ( ! redRecHitsEE.isValid() ) {
    edm::LogWarning("EERecoSummary") << "redRecHitsEE not found"; 
  }
  
  for ( EcalRecHitCollection::const_iterator itr = theEndcapEcalredRecHits->begin () ;
        itr != theEndcapEcalredRecHits->end () ; ++itr)
  {
      
    EEDetId eeid( itr -> id() );

    h_redRecHits_EE_recoFlag->Fill( itr -> recoFlag() );

  }

  // --- REC HITS ------------------------------------------------------------------------------------- 
  
  // ... endcap
  edm::Handle<EcalRecHitCollection> recHitsEE;
  ev.getByToken( recHitCollection_EE_, recHitsEE );
  const EcalRecHitCollection* theEndcapEcalRecHits = recHitsEE.product () ;
  if ( ! recHitsEE.isValid() ) {
    edm::LogWarning("EERecoSummary") << "recHitsEE not found"; 
  }

  float maxRecHitEnergyEEP = -999.;
  float maxRecHitEnergyEEM = -999.;

  EEDetId eeid_MrecHitEEM;
  EEDetId eeid_MrecHitEEP;

  for ( EcalRecHitCollection::const_iterator itr = theEndcapEcalRecHits->begin () ;
	itr != theEndcapEcalRecHits->end () ; ++itr)
    {
      
      EEDetId eeid( itr -> id() );

      // EE+
      if ( eeid.zside() > 0 ){

        h_recHits_EE_recoFlag       -> Fill( itr -> recoFlag() );

	// max E rec hit
	if (itr -> energy() > maxRecHitEnergyEEP && 
	    !(eeid.ix()>=41 && eeid.ix()<=60 && eeid.iy()>=41 && eeid.iy()<=60) ) {
	  maxRecHitEnergyEEP = itr -> energy() ;
	}
	
	// only channels above noise
	if (  itr -> energy() > ethrEE_ ){
	  h_recHits_EEP_Chi2          -> Fill( itr -> chi2() );
          h_recHits_EEP_time          -> Fill( itr -> time() );
	}
      }
      
      // EE-
      if ( eeid.zside() < 0 ){
	
        h_recHits_EE_recoFlag       -> Fill( itr -> recoFlag() );
	
	// max E rec hit
	if (itr -> energy() > maxRecHitEnergyEEM && 
	    !(eeid.ix()>=41 && eeid.ix()<=60 && eeid.iy()>=41 && eeid.iy()<=60) ) {
	  maxRecHitEnergyEEM = itr -> energy() ;
	}
	
	// only channels above noise
	if (  itr -> energy() > ethrEE_ ) {
	  h_recHits_EEM_Chi2          -> Fill( itr -> chi2() );
          h_recHits_EEM_time          -> Fill( itr -> time() );
	}

      }
    } // end loop over EE rec hits

  // energy
  h_recHits_EEP_energyMax -> Fill( maxRecHitEnergyEEP );
  h_recHits_EEM_energyMax -> Fill( maxRecHitEnergyEEM );

  //--- BASIC CLUSTERS --------------------------------------------------------------

  // ... endcap
  edm::Handle<edm::View<reco::CaloCluster> > basicClusters_EE_h;
  if(ev.getByToken( basicClusterCollection_EE_, basicClusters_EE_h )){

    for (unsigned int icl = 0; icl < basicClusters_EE_h->size(); ++icl) {
    
      //Get the associated RecHits
      const std::vector<std::pair<DetId,float> > & hits= (*basicClusters_EE_h)[icl].hitsAndFractions();
      for (std::vector<std::pair<DetId,float> > ::const_iterator rh = hits.begin(); rh!=hits.end(); ++rh){
      
        EBRecHitCollection::const_iterator itrechit = theEndcapEcalRecHits->find((*rh).first);
        if (itrechit==theEndcapEcalRecHits->end()) continue;
        h_basicClusters_recHits_EE_recoFlag -> Fill ( itrechit -> recoFlag() );
      }
  
    }
  }
  else{
    //    edm::LogWarning("EERecoSummary") << "basicClusters_EE_h not found"; 
  }

  // Super Clusters
  // ... endcap
  edm::Handle<reco::SuperClusterCollection> superClusters_EE_h;
  ev.getByToken( superClusterCollection_EE_, superClusters_EE_h );
  const reco::SuperClusterCollection* theEndcapSuperClusters = superClusters_EE_h.product () ;
  if ( ! superClusters_EE_h.isValid() ) {
    edm::LogWarning("EERecoSummary") << "superClusters_EE_h not found"; 
  }

  for (reco::SuperClusterCollection::const_iterator itSC = theEndcapSuperClusters->begin(); 
       itSC != theEndcapSuperClusters->end(); ++itSC ) {

    double scEt = itSC -> energy() * sin(2.*atan( exp(- itSC->position().eta() )));

    if (scEt < scEtThrEE_ ) continue;

    h_superClusters_eta       -> Fill( itSC -> eta() );
    h_superClusters_EE_phi    -> Fill( itSC -> phi() );
    
    if  ( itSC -> z() > 0 ){
      h_superClusters_EEP_nBC    -> Fill( itSC -> clustersSize() );      
    }

    if  ( itSC -> z() < 0 ){
      h_superClusters_EEM_nBC    -> Fill( itSC -> clustersSize() );      
    }
  }

  //--------------------------------------------------------
 
}


// ------------ method called once each job just before starting event loop  ------------
void 
EERecoSummary::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EERecoSummary::endJob() 
{}


//define this as a plug-in
DEFINE_FWK_MODULE(EERecoSummary);
