#ifndef __Basic2DGenericPFlowClusterizer_H__
#define __Basic2DGenericPFlowClusterizer_H__

#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"

class Basic2DGenericPFlowClusterizer : public PFClusterBuilderBase {
  typedef Basic2DGenericPFlowClusterizer B2DGPF;
 public:
  Basic2DGenericPFlowClusterizer(const edm::ParameterSet& conf) :
    PFClusterBuilderBase(conf) { }
  virtual ~Basic2DGenericPFlowClusterizer() {}
  Basic2DGenericPFlowClusterizer(const B2DGPF&) = delete;
  B2DGPF& operator=(const B2DGPF&) = delete;

  void buildPFClusters(const reco::PFClusterCollection&,
		       reco::PFClusterCollection& outclus);

 private:    
  
};

#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderFactory.h"
DEFINE_EDM_PLUGIN(PFClusterBuilderFactory,
		  Basic2DGenericPFlowClusterizer,
		  "Basic2DGenericPFlowClusterizer");

#endif
