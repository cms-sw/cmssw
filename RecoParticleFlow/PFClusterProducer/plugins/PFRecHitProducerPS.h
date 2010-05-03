#ifndef RecoParticleFlow_PFClusterProducer_PFRecHitProducerPS_h_
#define RecoParticleFlow_PFClusterProducer_PFRecHitProducerPS_h_

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

/**\class PFRecHitProducerPS 
\brief Producer for particle flow rechits (PFRecHit) 

\author Colin Bernet
\date   february 2008
*/

class CaloSubdetectorTopology;
class CaloSubdetectorGeometry;
class DetId;


class PFRecHitProducerPS : public PFRecHitProducer {
 public:
  explicit PFRecHitProducerPS(const edm::ParameterSet&);
  ~PFRecHitProducerPS();
  

 private:


  /// gets PS rechits, 
  /// translate them to PFRecHits, which are stored in the rechits vector
  void createRecHits(std::vector<reco::PFRecHit>& rechits,
		     std::vector<reco::PFRecHit>& rechitsCleaned,
		     edm::Event&, const edm::EventSetup&);




  /// find and set the neighbours to a given rechit
  /// this works for ecal, hcal, ps
  ///COLIN remonter cette fonction dans la classe de base 
  void 
    findRecHitNeighbours( reco::PFRecHit& rh, 
			  const std::map<unsigned,unsigned >& sortedHits, 
			  const CaloSubdetectorTopology& barrelTopo,
			  const CaloSubdetectorGeometry& barrelGeom, 
			  const CaloSubdetectorTopology& endcapTopo,
			  const CaloSubdetectorGeometry& endcapGeom );
  
  // ----------member data ---------------------------
   
  edm::InputTag    inputTagEcalRecHitsES_;
};

#endif
