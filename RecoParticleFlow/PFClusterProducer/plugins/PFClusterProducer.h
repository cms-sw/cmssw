#ifndef RecoParticleFlow_PFClusterProducer_PFClusterProducer_h_
#define RecoParticleFlow_PFClusterProducer_PFClusterProducer_h_

// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterAlgo.h"

/**\class PFClusterProducer 
\brief Producer for particle flow  clusters (PFCluster). 

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

  // ----------member data ---------------------------

  /// clustering algorithm 
  PFClusterAlgo    clusterAlgo_;


  /// verbose ?
  bool   verbose_;
  
  // ----------access to event data
  edm::InputTag    inputTagPFRecHits_;
  //---ab
  //std::string    inputTagClusterCollectionName_;
  //---ab
};

#endif
