#ifndef Calibration_RegionalEcalClusterProducer_h
#define Calibration_RegionalEcalClusterProducer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoEcal/EgammaClusterProducers/interface/IslandClusterProducer.h"
#include "RecoEcal/EgammaClusterAlgos/interface/IslandClusterAlgo.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalEtaPhiRegion.h"

//
// class decleration
//

class RegionalEcalClusterProducer : public edm::EDProducer {
   public:
      explicit RegionalEcalClusterProducer(const edm::ParameterSet&);
      ~RegionalEcalClusterProducer();

   private:

      IslandClusterAlgo* islandAlg;
      ClusterShapeAlgo shapeAlgo_;
      
      bool searchAroundTrack_;
      double deltaEtaSearch_;
      double deltaPhiSearch_;
      bool useEndcap_;
      edm::InputTag EBRecHitCollectionLabel_;
      edm::InputTag EERecHitCollectionLabel_;
      edm::InputTag l1tausource_;
      edm::InputTag prodtracksource_;

      std::string barrelClusterCollection_;
      std::string endcapClusterCollection_;

      double barrelSeedThresh_;
      double endcapSeedThresh_;

      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      void clusterize(edm::Event &evt, const edm::EventSetup &es, edm::InputTag ebrh, edm::InputTag eerh, std::string cluCol, std::vector<EcalEtaPhiRegion> etaphiR, IslandClusterAlgo::EcalPart ec_p);

      // ----------member data ---------------------------
};

#endif
