#ifndef __PFlow2DClusterizerWithTime_H__
#define __PFlow2DClusterizerWithTime_H__

#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"

#include <unordered_map>

class PFlow2DClusterizerWithTime : public PFClusterBuilderBase {
  typedef PFlow2DClusterizerWithTime B2DGPF;
 public:
  PFlow2DClusterizerWithTime(const edm::ParameterSet& conf);
    
  virtual ~PFlow2DClusterizerWithTime() {}
  PFlow2DClusterizerWithTime(const B2DGPF&) = delete;
  B2DGPF& operator=(const B2DGPF&) = delete;

  void update(const edm::EventSetup& es) { 
    _positionCalc->update(es); 
    if( _allCellsPosCalc ) _allCellsPosCalc->update(es);
    if( _convergencePosCalc ) _convergencePosCalc->update(es);
  }

  void buildClusters(const reco::PFClusterCollection&,
		     const std::vector<bool>&,
		     reco::PFClusterCollection& outclus);

 private:  
  const unsigned _maxIterations;
  const double _stoppingTolerance;
  const double _showerSigma2;
  const double _timeSigma_eb;
  const double _timeSigma_ee;
  const bool _excludeOtherSeeds;
  const double _minFracTot;
  
  const std::unordered_map<std::string,int> _layerMap;
  std::unordered_map<int,double> _recHitEnergyNorms;
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

DEFINE_EDM_PLUGIN(PFClusterBuilderFactory,
		  PFlow2DClusterizerWithTime,
		  "PFlow2DClusterizerWithTime");

#endif
