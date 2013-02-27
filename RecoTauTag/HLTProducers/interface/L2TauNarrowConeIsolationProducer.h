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






class L2TauNarrowConeIsolationProducer : public edm::EDProducer {
   public:
      typedef reco::CaloJet CaloJet;
      typedef reco::CaloJetCollection CaloJetCollection;
      explicit L2TauNarrowConeIsolationProducer(const edm::ParameterSet&);
      ~L2TauNarrowConeIsolationProducer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;


      //retrieve towers and crystals around the jet
      math::PtEtaPhiELorentzVectorCollection getECALHits(const CaloJet&,const edm::Event&,const edm::EventSetup& iSetup);
      math::PtEtaPhiELorentzVectorCollection getHCALHits(const CaloJet&,const edm::Event&);  
     
      edm::InputTag l2CaloJets_;//label for the readout Collection
      edm::InputTag EBRecHits_;//Label for ECAL Barrel Hits
      edm::InputTag EERecHits_;//Label for ECAL EndCAP Hits
      edm::InputTag CaloTowers_;//Label for ECAL EndCAP Hits

      double associationRadius_; //Association Distance  for a tower/crystal

      //Thresholding
      double crystalThresholdE_;
      double crystalThresholdB_;
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

        struct CrystalPtComparator
      	{
      	  bool operator()( const math::PtEtaPhiELorentzVector v1, const math::PtEtaPhiELorentzVector v2) const
      	    {
      	      return v1.pt() > v2.pt(); 
      	    }
       	};

	CrystalPtComparator comparePt;

};

