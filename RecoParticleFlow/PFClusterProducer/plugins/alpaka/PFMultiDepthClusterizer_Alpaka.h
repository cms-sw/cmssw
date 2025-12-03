#ifndef PFClusterProducer_plugins_alpaka_PFMultiDepthClusterizer_Alpaka_h
#define PFClusterProducer_plugins_alpaka_PFMultiDepthClusterizer_Alpaka_h

// Alpaka includes
#include <alpaka/alpaka.hpp>

#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"

#include "DataFormats/ParticleFlowReco/interface/alpaka/PFClusterDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitFractionDeviceCollection.h"

#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringCCLabelsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringVarsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringEdgeVarsDeviceCollection.h"

#include "HeterogeneousCore/AlpakaMath/interface/deltaPhi.h"

#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterParams.h"

/**
 * @class PFMultiDepthClusterizer_Alpaka
 * @brief Alpaka clusterizer algorithm for multi-depth particle flow clusters.
 * 
 * This class manages the execution of the full multi-stage particle flow clustering pipeline
 * using Alpaka, including link building, adjacency graph construction,
 * connected component detection (ECL-CC), and postprocessing.
 */

namespace ALPAKA_ACCELERATOR_NAMESPACE::eclcc {

  void clusterize(Queue& queue,
                  reco::PFClusterDeviceCollection& outPFCluster,
                  reco::PFRecHitFractionDeviceCollection& outPFRecHitFracs,
                  const reco::PFClusterDeviceCollection& pfCluster,
                  const reco::PFRecHitFractionDeviceCollection& pfRecHitFracs,
                  const reco::PFRecHitDeviceCollection& pfRecHit,
                  const PFMultiDepthClusterParams* params,
                  const unsigned int nClusters);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::eclcc

#endif
