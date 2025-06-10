#ifndef PFClusterProducer_plugins_alpaka_PFMultiDepthClusterizer_Alpaka_h
#define PFClusterProducer_plugins_alpaka_PFMultiDepthClusterizer_Alpaka_h

#include <Eigen/Core>
#include <Eigen/Dense>

// Alpaka includes
#include <alpaka/alpaka.hpp>

#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/AlpakaCore/interface/MoveToDeviceCache.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFClusterSoAProducerKernel.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitTopologyRecord.h"

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

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PFMultiDepthClusterizer_Alpaka {
  public:
    PFMultiDepthClusterizer_Alpaka() {}
    PFMultiDepthClusterizer_Alpaka(const PFMultiDepthClusterizer_Alpaka&) = delete;
    PFMultiDepthClusterizer_Alpaka& operator=(const PFMultiDepthClusterizer_Alpaka&) = delete;

    void apply(Queue& queue,
               reco::PFClusterDeviceCollection& outPFCluster,
               reco::PFRecHitFractionDeviceCollection& outPFRecHitFracs,
               const reco::PFClusterDeviceCollection& pfCluster,
               const reco::PFRecHitFractionDeviceCollection& pfRecHitFracs,
               const reco::PFRecHitDeviceCollection& pfRecHit,
	       const PFMultiDepthClusterParams* params,
	       const int nClusters);
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
