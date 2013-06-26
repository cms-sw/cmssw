/*
L2 Tau Trigger Isolation Producer

Author: Michail Bachtis
University of Wisconsin-Madison
e-mail: bachtis@hep.wisc.edu
*/


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/LorentzVectorFwd.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/TauReco/interface/L2TauInfoAssociation.h"


//Calorimeter!!
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h" 
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"






class L2TauIsolationProducer : public edm::EDProducer {
   public:
      explicit L2TauIsolationProducer(const edm::ParameterSet&);
      ~L2TauIsolationProducer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;


      //Retrieve Calo Hits 
      math::PtEtaPhiELorentzVectorCollection getECALHits(const reco::CaloJet&,const edm::Event&,const edm::EventSetup& iSetup); 
      math::PtEtaPhiELorentzVectorCollection getHCALHits(const reco::CaloJet&);

      edm::InputTag l2CaloJets_;//label for the readout Collection
      edm::InputTag EBRecHits_;//Label for ECAL Barrel Hits
      edm::InputTag EERecHits_;//Label for ECAL EndCAP Hits


      //Thresholding
      double crystalThreshold_;
      double towerThreshold_;


      //Sub Algorithm Configuration Variables

      //ECALIsolation
      bool ECALIsolation_run_;

      double ECALIsolation_innerCone_;
      double ECALIsolation_outerCone_;

      //TowerIsolation
      bool TowerIsolation_run_;

      double TowerIsolation_innerCone_;
      double TowerIsolation_outerCone_;

      //ECALClustering
      bool ECALClustering_run_;
      double ECALClustering_clusterRadius_;

      

      


};

