#ifndef RecoParticleFlow_PFClusterProducer_h_
#define RecoParticleFlow_PFClusterProducer_h_

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"



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

class PFClusterProducer : public edm::EDProducer {
 public:
  explicit PFClusterProducer(const edm::ParameterSet&);
  ~PFClusterProducer();

  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  

 private:

  /// find and set the neighbours to a given rechit
  void 
    findRecHitNeighbours( reco::PFRecHit* rh, 
			  const std::map<unsigned,reco::PFRecHit* >& hits, 
			  const CaloSubdetectorTopology& barrelTopo,
			  const CaloSubdetectorGeometry& barrelGeom, 
			  const CaloSubdetectorTopology& endcapTopo,
			  const CaloSubdetectorGeometry& endcapGeom );
  
  reco::PFRecHit*  createHcalRecHit( const DetId& detid, 
				     double energy,
				     int layer,
				     const CaloSubdetectorGeometry* geom );

  // ----------member data ---------------------------

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

  /// ecal barrel threshold
  double threshEcalBarrel_;

  /// ecal barrel seed threshold
  double threshSeedEcalBarrel_;

  /// ecal endcap threshold
  double threshEcalEndcap_;

  /// ecal endcap seed threshold
  double threshSeedEcalEndcap_;



  /// ps threshold
  double threshPS_;

  /// ps seed threshold
  double threshSeedPS_;



  /// hcal barrel threshold
  double threshHcalBarrel_;

  /// hcal barrel seed threshold
  double threshSeedHcalBarrel_;

  /// hcal endcap threshold
  double threshHcalEndcap_;

  /// hcal endcap seed threshold
  double threshSeedHcalEndcap_;


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
