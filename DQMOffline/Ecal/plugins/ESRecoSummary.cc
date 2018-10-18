// -*- C++ -*-
//
// Package:    ESRecoSummary
// Class:      ESRecoSummary
// Original Author:  Martina Malberti
// 
// system include files
#include <memory>
#include <iostream>
#include <cmath>

// user include files
#include "DQMOffline/Ecal/interface/ESRecoSummary.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalTools.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalCleaningAlgo.h"

//
// constructors and destructor
//
ESRecoSummary::ESRecoSummary(const edm::ParameterSet& ps)
{
  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  //now do what ever initialization is needed
  esRecHitCollection_        = consumes<ESRecHitCollection>(ps.getParameter<edm::InputTag>("recHitCollection_ES"));
  esClusterCollectionX_      = consumes<reco::PreshowerClusterCollection>(ps.getParameter<edm::InputTag>("ClusterCollectionX_ES"));
  esClusterCollectionY_      = consumes<reco::PreshowerClusterCollection>(ps.getParameter<edm::InputTag>("ClusterCollectionY_ES"));

  superClusterCollection_EE_ = consumes<reco::SuperClusterCollection>(ps.getParameter<edm::InputTag>("superClusterCollection_EE"));
}

void
ESRecoSummary::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const&, edm::EventSetup const&)
{
    // Monitor Elements (ex THXD)
  iBooker.setCurrentFolder(prefixME_ + "/ESRecoSummary"); // to organise the histos in folders

     
  // Preshower ----------------------------------------------
  h_recHits_ES_energyMax      = iBooker.book1D("recHits_ES_energyMax","recHits_ES_energyMax",200,0.,0.01);
  h_recHits_ES_time           = iBooker.book1D("recHits_ES_time","recHits_ES_time",200,-100.,100.);

  h_esClusters_energy_plane1 = iBooker.book1D("esClusters_energy_plane1","esClusters_energy_plane1",200,0.,0.01);
  h_esClusters_energy_plane2 = iBooker.book1D("esClusters_energy_plane2","esClusters_energy_plane2",200,0.,0.01);
  h_esClusters_energy_ratio  = iBooker.book1D("esClusters_energy_ratio","esClusters_energy_ratio",200,0.,20.);
}

//
// member functions
//

// ------------ method called to for each event  ------------
void ESRecoSummary::analyze(const edm::Event& ev, const edm::EventSetup&)
{
  //Preshower RecHits
  edm::Handle<ESRecHitCollection> recHitsES;
  ev.getByToken (esRecHitCollection_, recHitsES) ;
  const ESRecHitCollection* thePreShowerRecHits = recHitsES.product () ;

  if ( ! recHitsES.isValid() ) {
    std::cerr << "ESRecoSummary::analyze --> recHitsES not found" << std::endl; 
  }

  float maxRecHitEnergyES = -999.;

  for (ESRecHitCollection::const_iterator esItr = thePreShowerRecHits->begin(); esItr != thePreShowerRecHits->end(); ++esItr) 
    {
      
      h_recHits_ES_time   -> Fill(esItr->time()); 
      if (esItr -> energy() > maxRecHitEnergyES ) maxRecHitEnergyES = esItr -> energy() ;

    } // end loop over ES rec Hits

  h_recHits_ES_energyMax -> Fill(maxRecHitEnergyES ); 

  // ES clusters in X plane
  edm::Handle<reco::PreshowerClusterCollection> esClustersX;
  ev.getByToken( esClusterCollectionX_, esClustersX);
  const reco::PreshowerClusterCollection *ESclustersX = esClustersX.product();

  // ES clusters in Y plane
  edm::Handle<reco::PreshowerClusterCollection> esClustersY;
  ev.getByToken( esClusterCollectionY_, esClustersY);
  const reco::PreshowerClusterCollection *ESclustersY = esClustersY.product(); 
  

  // ... endcap
  edm::Handle<reco::SuperClusterCollection> superClusters_EE_h;
  ev.getByToken( superClusterCollection_EE_, superClusters_EE_h );
  const reco::SuperClusterCollection* theEndcapSuperClusters = superClusters_EE_h.product () ;
  if ( ! superClusters_EE_h.isValid() ) {
    std::cerr << "EcalRecHitSummary::analyze --> superClusters_EE_h not found" << std::endl; 
  }

  // loop over all super clusters
  for (reco::SuperClusterCollection::const_iterator itSC = theEndcapSuperClusters->begin(); 
       itSC != theEndcapSuperClusters->end(); ++itSC ) {
    
    if ( fabs(itSC->eta()) < 1.65 || fabs(itSC->eta()) > 2.6 ) continue;
    
    float ESenergyPlane1 = 0.;
    float ESenergyPlane2 = 0.;


    // Loop over all ECAL Basic clusters in the supercluster
    for (reco::CaloCluster_iterator ecalBasicCluster = itSC->clustersBegin(); ecalBasicCluster!= itSC->clustersEnd(); 
	 ecalBasicCluster++) {
      const reco::CaloClusterPtr ecalBasicClusterPtr = *(ecalBasicCluster);
      
      for (reco::PreshowerClusterCollection::const_iterator iESClus = ESclustersX->begin(); iESClus != ESclustersX->end(); 
	   ++iESClus) {
        const reco::CaloClusterPtr preshBasicCluster = iESClus->basicCluster();
        const reco::PreshowerCluster *esCluster = &*iESClus;
        if (preshBasicCluster == ecalBasicClusterPtr) {
	  ESenergyPlane1 += esCluster->energy();
	}
      }  // end of x loop
      
      for (reco::PreshowerClusterCollection::const_iterator iESClus = ESclustersY->begin(); iESClus != ESclustersY->end(); 
	   ++iESClus) {
        const reco::CaloClusterPtr preshBasicCluster = iESClus->basicCluster();
        const reco::PreshowerCluster *esCluster = &*iESClus;
        if (preshBasicCluster == ecalBasicClusterPtr) {
	  ESenergyPlane2 += esCluster->energy();
	}
      } // end of y loop
    } // end loop over all basic clusters in the supercluster

    //cout<<"DQM : "<<ESenergyPlane1<<" "<<ESenergyPlane2<<endl;
    h_esClusters_energy_plane1->Fill(ESenergyPlane1);
    h_esClusters_energy_plane2->Fill(ESenergyPlane2);
    if (ESenergyPlane1 > 0 && ESenergyPlane2 > 0) h_esClusters_energy_ratio -> Fill(ESenergyPlane1/ESenergyPlane2);
      
  }// end loop over superclusters

}

//define this as a plug-in
DEFINE_FWK_MODULE(ESRecoSummary);
