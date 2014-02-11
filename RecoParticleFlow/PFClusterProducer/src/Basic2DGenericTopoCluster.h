#ifndef __Basic2DGenericTopoCluster_H__
#define __Basic2DGenericTopoCluster_H__

#include "RecoParticleFlow/PFClusterProducer/interface/TopoClusterBuilderBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"

class Basic2DGenericTopoCluster : public TopoClusterBuilderBase {
  typedef Basic2DGenericTopoCluster B2DGT;
 public:
  Basic2DGenericTopoCluster(const edm::ParameterSet& conf) :
    TopoClusterBuilderBase(conf),
    _useCornerCells(conf.getParameter<bool>("useCornerCells")) { }
  virtual ~Basic2DGenericTopoCluster() {}
  Basic2DGenericTopoCluster(const B2DGT&) = delete;
  B2DGT& operator=(const B2DGT&) = delete;

  void buildTopoClusters(const reco::PFRecHitRefVector&,
			 const std::vector<bool>&,
			 reco::PFClusterCollection&);

 private:  
  const bool _useCornerCells;
  void buildTopoCluster(const reco::PFRecHitRefVector&, // hits to cluster
			const std::vector<bool>&, // masked rechits
			std::vector<bool>&, // hit usage state
			const reco::PFRecHitRef&, //present rechit
			reco::PFCluster&); // the topocluster
  
};

#include "RecoParticleFlow/PFClusterProducer/interface/TopoClusterBuilderFactory.h"
DEFINE_EDM_PLUGIN(TopoClusterBuilderFactory,
		  Basic2DGenericTopoCluster,"Basic2DGenericTopoCluster");

#endif
