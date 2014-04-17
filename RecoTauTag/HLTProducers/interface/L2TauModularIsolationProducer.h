/*
L2 Tau Trigger Isolation Producer

Author: Michail Bachtis
University of Wisconsin-Madison
e-mail: bachtis@hep.wisc.edu
*/

// system include files
#include <memory>
#include <string>
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
#include "Geometry/Records/interface/CaloGeometryRecord.h"

//PF
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"


// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"



class L2TauModularIsolationProducer : public edm::EDProducer {
   public:
      explicit L2TauModularIsolationProducer(const edm::ParameterSet&);
      ~L2TauModularIsolationProducer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob();

      //retrieve towers / crystals / clusters around the jet
      math::PtEtaPhiELorentzVectorCollection getECALHits(const reco::CaloJet&,const edm::Event&,const edm::EventSetup& iSetup);
      math::PtEtaPhiELorentzVectorCollection getHCALHits(const reco::CaloJet&,const edm::Event&);  
      math::PtEtaPhiELorentzVectorCollection getPFClusters(const reco::CaloJet&,const edm::Event&,const edm::EDGetTokenT<reco::PFClusterCollection>&);  

      edm::EDGetTokenT<reco::CaloJetCollection> l2CaloJets_;//label for the readout Collection
      edm::EDGetTokenT<EBRecHitCollection> EBRecHits_;//Label for ECAL Barrel Hits
      edm::EDGetTokenT<EERecHitCollection> EERecHits_;//Label for ECAL EndCAP Hits
      edm::EDGetTokenT<CaloTowerCollection> caloTowers_;//Label for ECAL EndCAP Hits
      edm::EDGetTokenT<reco::PFClusterCollection> pfClustersECAL_;//Label for ECAL PF Clusters
      edm::EDGetTokenT<reco::PFClusterCollection> pfClustersHCAL_;//Label for HCAL PF Clusters

      //Algorithm Configuration Variables
      std::string ecalIsolationAlg_;
      std::string hcalIsolationAlg_;
      std::string ecalClusteringAlg_;
      std::string hcalClusteringAlg_;

      double associationRadius_;

      double simpleClusterRadiusECAL_;
      double simpleClusterRadiusHCAL_;
      double innerConeECAL_;
      double outerConeECAL_;
      double innerConeHCAL_;
      double outerConeHCAL_;

      //Thresholding
      double crystalThresholdE_;
      double crystalThresholdB_;
      double towerThreshold_;


        struct RecHitPtComparator
      	{
      	  bool operator()( const math::PtEtaPhiELorentzVector& v1, const math::PtEtaPhiELorentzVector& v2) const
      	    {
      	      return v1.pt() > v2.pt(); 
      	    }
       	};

	RecHitPtComparator comparePt;
      



};

