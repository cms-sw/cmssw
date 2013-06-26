#ifndef RecoParticleFlow_PFClusterProducer_PFRecHitProducerECAL_h_
#define RecoParticleFlow_PFClusterProducer_PFRecHitProducerECAL_h_

// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CaloTopology/interface/CaloDirection.h"

#include "RecoParticleFlow/PFClusterProducer/plugins/PFRecHitProducer.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"

#include "DataFormats/Math/interface/Vector3D.h"
/**\class PFRecHitProducerECAL 
\brief Producer for particle flow rechits (PFRecHit) in ECAL

\author Colin Bernet
\date   february 2008
*/

class CaloSubdetectorTopology;
class CaloSubdetectorGeometry;
class EcalBarrelGeometry;
class EcalEndcapGeometry;
class CaloSubdetectorGeometry;
class DetId;





class PFRecHitProducerECAL : public PFRecHitProducer {

 public:
  explicit PFRecHitProducerECAL(const edm::ParameterSet&);
  ~PFRecHitProducerECAL();

  

 private:

  /// gets ecal barrel and endcap rechits, 
  /// translate them to PFRecHits, which are stored in the rechits vector
  void createRecHits(std::vector<reco::PFRecHit>& rechits,
		     std::vector<reco::PFRecHit>& rechitsCleaned,
		     edm::Event&, const edm::EventSetup&);


  

  reco::PFRecHit*  createEcalRecHit( const DetId& detid,
				     double energy,
				     PFLayer::Layer layer,
				     const CaloSubdetectorGeometry* geom );



  /// find the position and the axis of the cell for a given rechit
  bool findEcalRecHitGeometry ( const DetId& detid, 
				const CaloSubdetectorGeometry* geom,
				math::XYZVector& position, 
				math::XYZVector& axis );

  /// find rechit neighbours, using the hashed index
  void 
    findRecHitNeighboursECAL( reco::PFRecHit& rh, 
			      const std::map<unsigned,unsigned >& sortedHits );

  /// fill the vectors neighboursEB_ and neighboursEE_ 
  /// which keep track of the neighbours of each rechit. 
  /// to be called at the beginning of the run
  void ecalNeighbArray( const EcalBarrelGeometry& barrelGeom,
			const CaloSubdetectorTopology& barrelTopo,
			const EcalEndcapGeometry& endcapGeom,
			const CaloSubdetectorTopology& endcapTopo );

  DetId move(DetId cell, const CaloDirection& dir ) const;

  bool stdsimplemove(DetId& cell, 
		     const CaloDirection& dir,
		     const CaloSubdetectorTopology& barrelTopo,
		     const CaloSubdetectorTopology& endcapTopo,
		     const EcalBarrelGeometry& barrelGeom,
		     const EcalEndcapGeometry& endcapGeom ) const;


  bool stdmove(DetId& cell, 
	       const CaloDirection& dir,
	       const CaloSubdetectorTopology& barrelTopo,
	       const CaloSubdetectorTopology& endcapTopo,
	       const EcalBarrelGeometry& barrelGeom,
	       const EcalEndcapGeometry& endcapGeom ) const;

  // ----------member data ---------------------------
  

 
  /// for each ecal barrel rechit, keep track of the neighbours
  std::vector<std::vector<DetId> >  neighboursEB_;

  /// for each ecal endcap rechit, keep track of the neighbours
  std::vector<std::vector<DetId> >  neighboursEE_;
  
  /// set to true in ecalNeighbArray
  bool  neighbourmapcalculated_;

  /// if true, navigation will cross the barrel-endcap border
  bool  crossBarrelEndcapBorder_;

  // ----------access to event data
  edm::InputTag    inputTagEcalRecHitsEB_;
  edm::InputTag    inputTagEcalRecHitsEE_;

  // ---- Perform timing cleaning
  bool timingCleaning_;
  double threshCleaningEB_;
  double threshCleaningEE_;

  // ---- Perform topological cleaning
  bool topologicalCleaning_;

};

#endif
