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

  /// process ECAL rechits
  void produceEcal(edm::Event&, const edm::EventSetup&);

  /// process HCAL rechits
  void produceHcal(edm::Event&, const edm::EventSetup&);

  reco::PFRecHit*  createEcalRecHit( const DetId& detid,
				     double energy,
				     int layer,
				     const CaloSubdetectorGeometry* geom );

  reco::PFRecHit*  createHcalRecHit( const DetId& detid, 
				     double energy,
				     int layer,
				     const CaloSubdetectorGeometry* geom );
  
  /* not necessary to have a function for this: */
  /*   reco::PFRecHit*  createHcalCTRecHit( const DetId& detid,  */
  /* 				       double energy, */
  /* 				       int layer, */
  /* 				       const CaloSubdetectorGeometry* geom ); */

  



  /// find the position and the axis of the cell for a given rechit
  bool findEcalRecHitGeometry ( const DetId& detid, 
				const CaloSubdetectorGeometry* geom,
				math::XYZVector& position, 
				math::XYZVector& axis );


  /// find and set the neighbours to a given rechit
  void 
    findRecHitNeighbours( reco::PFRecHit* rh, 
			  const std::map<unsigned,reco::PFRecHit* >& hits, 
			  const CaloSubdetectorTopology& barrelTopo,
			  const CaloSubdetectorGeometry& barrelGeom, 
			  const CaloSubdetectorTopology& endcapTopo,
			  const CaloSubdetectorGeometry& endcapGeom );
  
  void 
    findRecHitNeighboursCT( reco::PFRecHit* rh, 
			    const std::map<unsigned,reco::PFRecHit* >& rechits,
			    const CaloSubdetectorTopology& topology, 
			    const CaloSubdetectorGeometry& geometry );
  
  DetId getNorth(const DetId& id, const CaloSubdetectorTopology& topology);
  DetId getSouth(const DetId& id, const CaloSubdetectorTopology& topology);
  

  

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

  /// ecal number of neighbours
  int nNeighboursEcal_;
  
  /// ecal number of crystals for position calculation
  int posCalcNCrystalEcal_;

  /// ecal parameter for position calculation  
  double posCalcP1Ecal_;
  
  /// ecal sigma of shower (cm)
  double showerSigmaEcal_;


  // PS parameters ===============================================

  /// ps threshold
  double threshPS_;

  /// ps seed threshold
  double threshSeedPS_;

  /// ps number of neighbours
  int nNeighboursPS_;
  
  /// ps number of crystals for position calculation
  int posCalcNCrystalPS_;

  /// ps parameter for position calculation  
  double posCalcP1PS_;
  
  /// ps sigma of shower (cm)
  double showerSigmaPS_;


  // HCAL parameters =============================================


  /// hcal barrel threshold
  double threshHcalBarrel_;

  /// hcal barrel seed threshold
  double threshSeedHcalBarrel_;

  /// hcal endcap threshold
  double threshHcalEndcap_;

  /// hcal endcap seed threshold
  double threshSeedHcalEndcap_;

  /// Hcal number of neighbours
  int nNeighboursHcal_;
  
  /// Hcal number of crystals for position calculation
  int posCalcNCrystalHcal_;

  /// Hcal parameter for position calculation  
  double posCalcP1Hcal_;
  
  /// Hcal sigma of shower (cm)
  double showerSigmaHcal_;



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
