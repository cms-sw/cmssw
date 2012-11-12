#ifndef RecoParticleFlow_PFClusterProducer_PFRecHitProducerHO_h_
#define RecoParticleFlow_PFClusterProducer_PFRecHitProducerHO_h_

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
/**\class PFRecHitProducerHO 
\brief Producer for particle flow rechits (PFRecHit) in HO

\author Gobinda Majumder
\date   November 2011
*/

class CaloSubdetectorTopology;
class HcalTopology;
class CaloSubdetectorGeometry;
class DetId;

class PFRecHitProducerHO : public PFRecHitProducer {

 public:
  explicit PFRecHitProducerHO(const edm::ParameterSet&);
  ~PFRecHitProducerHO();

 private:

  // gets HO rechits, 
  // translate them to PFRecHits, which are stored in the rechits vector
  void createRecHits(std::vector<reco::PFRecHit>& rechits,
		     std::vector<reco::PFRecHit>& rechitsCleaned,
		     edm::Event&, const edm::EventSetup&);

  reco::PFRecHit*  createHORecHit( const DetId& detid,
				     double energy,
				     PFLayer::Layer layer,
				   const CaloSubdetectorGeometry* geom);
  //				     unsigned newDetId=0);



  /// find the position and the axis of the cell for a given rechit
  bool findHORecHitGeometry ( const DetId& detid, 
				const CaloSubdetectorGeometry* geom,
				math::XYZVector& position, 
				math::XYZVector& axis );

  /// find rechit neighbours, using the hashed index
    void 
      findRecHitNeighboursHO( reco::PFRecHit& rh, const HcalTopology& topo,
  			      const std::map<unsigned,unsigned >& sortedHits );
void
    findRecHitNeighbours( reco::PFRecHit& rh, 
			  const std::map<unsigned,unsigned >& sortedHits, 
			  const CaloSubdetectorTopology& barrelTopo,
			  const CaloSubdetectorGeometry& barrelGeom); 

  /// fill the vectors neighboursEB_ and neighboursEE_ 
  /// which keep track of the neighbours of each rechit. 
  /// to be called at the beginning of the run
  void hoNeighbArray( const CaloSubdetectorGeometry& barrelGeom,
			const HcalTopology& barrelTopo);

  DetId move(DetId cell, const HcalTopology& barrelTopo, const CaloDirection& dir ) const;

  bool stdsimplemove(DetId& cell, 
		     const CaloDirection& dir,
		     const CaloSubdetectorTopology& barrelTopo,
		     const CaloSubdetectorGeometry& barrelGeom) const;

  bool stdmove(DetId& cell, 
	       const CaloDirection& dir,
	       const CaloSubdetectorTopology& barrelTopo,
	       const CaloSubdetectorGeometry& barrelGeom) const; 

  // ----------member data ---------------------------
  

 
  /// for each HO barrel rechit, keep track of the neighbours
  std::vector<std::vector<DetId> >  neighboursHO_;

  /// set to true in hoNeighbArray
  bool  neighbourmapcalculated_;

//  // if true, navigation will cross the barrel-endcap border
//  bool  crossBarrelEndcapBorder_;

  // ----------access to event data
  edm::InputTag    inputTagHORecHits_;

  // ---- Thresholds on Ring0 hit energies
  double thresholdR0_;

  // --- Thresholds on Rings +/-1, +/-2 hit energies
  double thresholdR1_;

  // Maximum allowed severity of HO rechits.  Hits above the given severity level will be rejected.  Default max value is 9 (the same as used for accepting hits in the default Hcal caloTowers)
  int HOMaxAllowedSev_;
};

#endif
