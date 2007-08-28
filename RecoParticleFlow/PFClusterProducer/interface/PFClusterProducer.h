#ifndef RecoParticleFlow_PFClusterProducer_h_
#define RecoParticleFlow_PFClusterProducer_h_

// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
// #include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "RecoParticleFlow/PFClusterAlgo/interface/PFClusterAlgo.h"

/**\class PFClusterProducer 
\brief Producer for particle flow rechits (PFRecHit) 
and clusters (PFCluster). 

This producer makes use of PFClusterAlgo, the clustering algorithm 
for particle flow clusters.

\author Colin Bernet
\date   July 2006
*/

class CaloSubdetectorTopology;
class CaloSubdetectorGeometry;
class DetId;

namespace reco {
  class PFRecHit;
}


class PFClusterProducer : public edm::EDProducer {
 public:
  explicit PFClusterProducer(const edm::ParameterSet&);
  ~PFClusterProducer();

  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  

 private:

  /// gets ecal barrel and endcap rechits, 
  /// translate them to PFRecHits, which are stored in the rechits vector
  void createEcalRecHits(std::vector<reco::PFRecHit>& rechits,
			 edm::Event&, const edm::EventSetup&);

  /// gets hcal barrel and endcap rechits, 
  /// translate them to PFRecHits, which are stored in the rechits vector
  void createHcalRecHits(std::vector<reco::PFRecHit>& rechits,
			 edm::Event&, const edm::EventSetup&);



  /// gets PS rechits, 
  /// translate them to PFRecHits, which are stored in the rechits vector
  void createPSRecHits(std::vector<reco::PFRecHit>& rechits,
		       edm::Event&, const edm::EventSetup&);





  reco::PFRecHit*  createEcalRecHit( const DetId& detid,
				     double energy,
				     int layer,
				     const CaloSubdetectorGeometry* geom );

  reco::PFRecHit*  createHcalRecHit( const DetId& detid, 
				     double energy,
				     int layer,
				     const CaloSubdetectorGeometry* geom,
				     unsigned newDetId=0);
  

  


  /// find the position and the axis of the cell for a given rechit
  bool findEcalRecHitGeometry ( const DetId& detid, 
				const CaloSubdetectorGeometry* geom,
				math::XYZVector& position, 
				math::XYZVector& axis );


  /// find and set the neighbours to a given rechit
  /// this works for ecal, hcal, ps
  void 
    findRecHitNeighbours( reco::PFRecHit& rh, 
			  const std::map<unsigned,unsigned >& sortedHits, 
			  const CaloSubdetectorTopology& barrelTopo,
			  const CaloSubdetectorGeometry& barrelGeom, 
			  const CaloSubdetectorTopology& endcapTopo,
			  const CaloSubdetectorGeometry& endcapGeom );
  
  /// find and set the neighbours to a given rechit
  /// this works for hcal CaloTowers. 
  /// Should be possible to have a single function for all detectors
  void 
    findRecHitNeighboursCT( reco::PFRecHit& rh, 
			    const std::map<unsigned,unsigned >& sortedHits,
			    const CaloSubdetectorTopology& topology );
  
  DetId getNorth(const DetId& id, const CaloSubdetectorTopology& topology);
  DetId getSouth(const DetId& id, const CaloSubdetectorTopology& topology);
  

  

  // ----------member data ---------------------------

  /// clustering algorithm for ECAL
  PFClusterAlgo    clusterAlgoECAL_;
  
  /// clustering algorithm for HCAL
  PFClusterAlgo    clusterAlgoHCAL_;
  
  /// clustering algorithm for PS
  PFClusterAlgo    clusterAlgoPS_;
  

  /// process Ecal ? 
  bool   processEcal_;

  /// process Hcal ? 
  bool   processHcal_;  

  /// process preshower ? 
  bool   processPS_;  


  /// clustering Ecal ? 
  bool   clusteringEcal_;

  /// clustering Hcal ? 
  bool   clusteringHcal_;  

  /// clustering Hcal: use CaloTowers for navigation ? 
  bool   clusteringHcalCaloTowers_;  

  /// clustering preshower ? 
  bool   clusteringPS_;  

  /// verbose ?
  bool   verbose_;


  /// produce rechits yes/no 
  bool   produceRecHits_;
  

  // ----------access to event data
  std::string ecalRecHitsEBModuleLabel_;
  std::string ecalRecHitsEBProductInstanceName_;
  std::string ecalRecHitsEEModuleLabel_;
  std::string ecalRecHitsEEProductInstanceName_;
  std::string ecalRecHitsESModuleLabel_;
  std::string ecalRecHitsESProductInstanceName_;
  std::string hcalRecHitsHBHEModuleLabel_;
  std::string hcalRecHitsHBHEProductInstanceName_;
  std::string caloTowersModuleLabel_;
  std::string caloTowersProductInstanceName_;
};

#endif
