#ifndef __GenericTopoClusterizer_H__
#define __GenericTopoClusterizer_H__

#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"

class GenericTopoClusterizer : public InitialClusteringStepBase {
  typedef GenericTopoClusterizer B2DGT;
 public:
 GenericTopoClusterizer(const edm::ParameterSet& conf,
			edm::ConsumesCollector& sumes) :
  InitialClusteringStepBase(conf,sumes){ }
  virtual ~GenericTopoClusterizer() {}
  GenericTopoClusterizer(const B2DGT&) = delete;
  B2DGT& operator=(const B2DGT&) = delete;

  void buildClusters(const edm::Handle<reco::PFRecHitCollection>&,
		     const std::vector<bool>&,
		     const std::vector<bool>&, 
		     reco::PFClusterCollection&);
  
 private:  
  void buildTopoCluster(const edm::Handle<reco::PFRecHitCollection>&,
			const std::vector<bool>&, // masked rechits
			const reco::PFRecHitRef&, //present rechit
			std::vector<bool>&, // hit usage state
			reco::PFCluster&); // the topocluster
  
};

DEFINE_EDM_PLUGIN(InitialClusteringStepFactory,
		  GenericTopoClusterizer,
		  "GenericTopoClusterizer");

#endif
