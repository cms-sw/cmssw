#ifndef __Semi3DArborClusterizerWithSharing_H__
#define __Semi3DArborClusterizerWithSharing_H__

#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"

#include <unordered_map>

#include "RecoParticleFlow/PFClusterProducer/interface/Arbor.hh"

class Semi3DArborClusterizerWithSharing : public PFClusterBuilderBase {
  typedef Semi3DArborClusterizerWithSharing S3DACWS;
 public:
  Semi3DArborClusterizerWithSharing(const edm::ParameterSet& conf);
    
  virtual ~Semi3DArborClusterizerWithSharing() {}
  Semi3DArborClusterizerWithSharing(const S3DACWS&) = delete;
  S3DACWS& operator=(const S3DACWS&) = delete;

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
		  Semi3DArborClusterizerWithSharing,
		  "Semi3DArborClusterizerWithSharing");

#endif
