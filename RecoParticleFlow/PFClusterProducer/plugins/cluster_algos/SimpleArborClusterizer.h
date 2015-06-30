#ifndef __SimpleArborClusterizer_H__
#define __SimpleArborClusterizer_H__

#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"

class SimpleArborClusterizer : public InitialClusteringStepBase {
  typedef SimpleArborClusterizer B2DGT;
 public:
 SimpleArborClusterizer(const edm::ParameterSet& conf,
			edm::ConsumesCollector& sumes) :
  InitialClusteringStepBase(conf,sumes),
      _cellSize(conf.getParameter<double>("cellSize")),
      _layerThickness(conf.getParameter<double>("layerThickness")),
      _distSeedForMerge(conf.getParameter<double>("distSeedForMerge")),      
      _killNoiseClusters(conf.getParameter<bool>("killNoiseClusters")),
      _allowSameLayerSeedMerge(conf.getParameter<bool>("allowSameLayerSeedMerge")),
      _maxNoiseClusterSize(conf.getParameter<unsigned>("maxNoiseClusterSize")) { }
  virtual ~SimpleArborClusterizer() {}
  SimpleArborClusterizer(const B2DGT&) = delete;
  B2DGT& operator=(const B2DGT&) = delete;

  void buildClusters(const edm::Handle<reco::PFRecHitCollection>&,
		     const std::vector<bool>&,
		     const std::vector<bool>&, 
		     reco::PFClusterCollection&);
  
 private:  
  double _cellSize,_layerThickness,_distSeedForMerge;
  bool _killNoiseClusters,_allowSameLayerSeedMerge;
  unsigned _maxNoiseClusterSize;
};

DEFINE_EDM_PLUGIN(InitialClusteringStepFactory,
		  SimpleArborClusterizer,
		  "SimpleArborClusterizer");

#endif
