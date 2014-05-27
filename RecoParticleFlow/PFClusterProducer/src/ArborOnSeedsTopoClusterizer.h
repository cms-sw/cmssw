#ifndef __ArborOnSeedsTopoClusterizer_H__
#define __ArborOnSeedsTopoClusterizer_H__

#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "RecoParticleFlow/PFClusterProducer/interface/Arbor.hh"

class ArborOnSeedsTopoClusterizer : public InitialClusteringStepBase {
  typedef ArborOnSeedsTopoClusterizer B2DGT;
 public:
  ArborOnSeedsTopoClusterizer(const edm::ParameterSet& conf) :
    InitialClusteringStepBase(conf),
      _useCornerCells(conf.getParameter<bool>("useCornerCells")),
      _showerSigma(conf.getParameter<double>("showerSigma")) { }
  virtual ~ArborOnSeedsTopoClusterizer() {}
  ArborOnSeedsTopoClusterizer(const B2DGT&) = delete;
  B2DGT& operator=(const B2DGT&) = delete;

  void buildClusters(const edm::Handle<reco::PFRecHitCollection>&,
		     const std::vector<bool>&,
		     const std::vector<bool>&, 
		     reco::PFClusterCollection&);
  
 private:  
  const bool _useCornerCells;
  const double _showerSigma;
  
  

  void arborizeSeeds(const reco::PFRecHitCollection&,
		     const std::vector<std::pair<unsigned,double> >&,
		     arbor::branchcoll&) const;
  
  void buildTopoCluster(const edm::Handle<reco::PFRecHitCollection>&,
			const std::vector<bool>&, // masked rechits
			const reco::PFRecHitRef&, //present rechit
			std::vector<bool>&, // hit usage state
			reco::PFCluster&); // the topocluster

  void getLinkedTopoClusters(const std::unordered_multimap<unsigned,unsigned>&,
			     const std::unordered_multimap<unsigned,unsigned>&,
			     const reco::PFClusterCollection&,
			     const unsigned,
			     std::vector<bool>&,
			     std::vector<unsigned>&) const;

  /*
  void seedPFClustersFromTopo(const reco::PFCluster&,
			      const std::vector<bool>&,
			      reco::PFClusterCollection&) const;

  void growPFClusters(const std::unordered_multimap<unsigned,unsigned>&,
		      const reco::PFClusterCollection& topoclusters,
		      const unsigned,
		      double,
		      reco::PFClusterCollection&);
  */

  
  
};

DEFINE_EDM_PLUGIN(InitialClusteringStepFactory,
		  ArborOnSeedsTopoClusterizer,
		  "ArborOnSeedsTopoClusterizer");

#endif
