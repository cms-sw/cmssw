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
    _allCellsPosCalc.reset(NULL);
    if( conf.exists("allCellsPositionCalc") ) {
      const edm::ParameterSet& acConf = 
	conf.getParameterSet("allCellsPositionCalc");
      const std::string& algoac = 
	acConf.getParameter<std::string>("algoName");
      PosCalc* accalc = 
	PFCPositionCalculatorFactory::get()->create(algoac, acConf);
      _allCellsPosCalc.reset(accalc);
    }
    // if necessary a third pos calc for convergence testing
    _convergencePosCalc.reset(NULL);
    if( conf.exists("positionCalcForConvergence") ) {
      const edm::ParameterSet& convConf = 
	conf.getParameterSet("positionCalcForConvergence");
      const std::string& algoconv = 
	convConf.getParameter<std::string>("algoName");
      PosCalc* convcalc = 
	PFCPositionCalculatorFactory::get()->create(algoconv, convConf);
      _convergencePosCalc.reset(convcalc);
    }
  }
  virtual ~Basic2DGenericPFlowClusterizer() {}
  Basic2DGenericPFlowClusterizer(const B2DGPF&) = delete;
  B2DGPF& operator=(const B2DGPF&) = delete;

  void update(const edm::EventSetup& es) { 
    _positionCalc->update(es); 
    if( _allCellsPosCalc ) _allCellsPosCalc->update(es);
    if( _convergencePosCalc ) _convergencePosCalc->update(es);
  }

  void buildPFClusters(const reco::PFClusterCollection&,
		       const std::vector<bool>&,
		       reco::PFClusterCollection& outclus);

 private:  
  const unsigned _maxIterations;
  const double _stoppingTolerance;
  const double _showerSigma;
  const bool _excludeOtherSeeds;
  std::unique_ptr<PFCPositionCalculatorBase> _allCellsPosCalc;
  std::unique_ptr<PFCPositionCalculatorBase> _convergencePosCalc;
  
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
