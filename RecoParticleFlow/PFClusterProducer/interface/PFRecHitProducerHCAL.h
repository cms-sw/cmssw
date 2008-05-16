#ifndef RecoParticleFlow_PFRecHitProducerHCAL_h_
#define RecoParticleFlow_PFRecHitProducerHCAL_h_

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
#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitProducer.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"

/**\class PFRecHitProducerHCAL
\brief Producer for particle flow rechits  (PFRecHit) in HCAL 

\author Colin Bernet
\date   february 2008
*/

class CaloSubdetectorTopology;
class CaloSubdetectorGeometry;
class DetId;


class PFRecHitProducerHCAL : public PFRecHitProducer {
 public:
  explicit PFRecHitProducerHCAL(const edm::ParameterSet&);
  ~PFRecHitProducerHCAL();

 
 private:


  /// gets hcal barrel and endcap rechits, 
  /// translate them to PFRecHits, which are stored in the rechits vector
  void createRecHits(std::vector<reco::PFRecHit>& rechits,
		     edm::Event&, const edm::EventSetup&);




  reco::PFRecHit*  createHcalRecHit( const DetId& detid, 
				     double energy,
				     PFLayer::Layer layer,
				     const CaloSubdetectorGeometry* geom,
				     unsigned newDetId=0);
  

  

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
  
  // ----------access to event data
  edm::InputTag    inputTagHcalRecHitsHBHE_;
  edm::InputTag    inputTagCaloTowers_;
  
  /// threshold for HF
  double           thresh_HF_;
};

#endif
