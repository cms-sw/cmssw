#ifndef __Basic2DGenericPFlowClusterizer_H__
#define __Basic2DGenericPFlowClusterizer_H__

#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"

class Basic2DGenericPFlowClusterizer : public PFClusterBuilderBase {
  typedef Basic2DGenericPFlowClusterizer B2DGPF;
 public:
  Basic2DGenericPFlowClusterizer(const edm::ParameterSet& conf) :
    PFClusterBuilderBase(conf),
    _maxIterations(conf.getParameter<unsigned>("maxIterations")),
    _stoppingTolerance(conf.getParameter<double>("stoppingTolerance")),
    _showerSigma(conf.getParameter<double>("showerSigma")),
    _excludeOtherSeeds(conf.getParameter<bool>("excludeOtherSeeds")) {    
    const edm::ParameterSet& pcConf = conf.getParameterSet("altPositionCalc");
    const std::string& algo = pcConf.getParameter<std::string>("algoName");
    PosCalc* calcp = PFCPositionCalculatorFactory::get()->create(algo, pcConf);
    _allCellsPosCalc.reset(calcp);
  }
  virtual ~Basic2DGenericPFlowClusterizer() {}
  Basic2DGenericPFlowClusterizer(const B2DGPF&) = delete;
  B2DGPF& operator=(const B2DGPF&) = delete;

  void buildPFClusters(const reco::PFClusterCollection&,
		       const std::vector<bool>&,
		       reco::PFClusterCollection& outclus);

 private:  
  const unsigned _maxIterations;
  const double _stoppingTolerance;
  const double _showerSigma;
  const bool _excludeOtherSeeds;
  std::unique_ptr<PFCPositionCalculatorBase> _allCellsPosCalc;
  
  void seedPFClustersFromTopo(const reco::PFCluster&,
			      const std::vector<bool>&,
			      reco::PFClusterCollection&) const;

  void growPFClusters(const reco::PFCluster&,
		      const std::vector<bool>&,
		      const unsigned toleranceScaling,
		      const unsigned iter,
		      double dist,
		      reco::PFClusterCollection&) const;
  void prunePFClusters(reco::PFClusterCollection&) const;
};

#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderFactory.h"
DEFINE_EDM_PLUGIN(PFClusterBuilderFactory,
		  Basic2DGenericPFlowClusterizer,
		  "Basic2DGenericPFlowClusterizer");

#endif
