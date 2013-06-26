#ifndef RecoParticleFlow_PFClusterProducer_PFHCALSuperClusterProducer_h_
#define RecoParticleFlow_PFClusterProducer_PFHCALSuperClusterProducer_h_

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
#include "DataFormats/ParticleFlowReco/interface/PFSuperClusterFwd.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFHcalSuperClusterAlgo.h"

/**\class PFHCALSuperClusterProducer 
\brief Producer for particle flow superclusters (PFSuperCluster). 

This producer makes use of PFHcalSuperClusterAlgo, the clustering algorithm 
for particle flow superclusters.

\author Chris Tully
\date   July 2012
*/

class CaloSubdetectorTopology;
class CaloSubdetectorGeometry;
class DetId;

namespace reco {
  class PFRecHit;
}


class PFHCALSuperClusterProducer : public edm::EDProducer {
 public:
  explicit PFHCALSuperClusterProducer(const edm::ParameterSet&);
  ~PFHCALSuperClusterProducer();

  
  virtual void produce(edm::Event&, const edm::EventSetup&);

  virtual void endJob();
  

 private:

  // ----------member data ---------------------------

  /// clustering algorithm 
  PFHcalSuperClusterAlgo    hcalSuperClusterAlgo_;


  /// verbose ?
  bool   verbose_;
  
  // ----------access to event data
  edm::InputTag    inputTagPFClusters_;
  edm::InputTag    inputTagPFClustersHO_;
};

#endif
