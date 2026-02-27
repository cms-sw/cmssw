#ifndef PFClusterProducer_plugins_alpaka_PFMultiDepthClusterizer_h
#define PFClusterProducer_plugins_alpaka_PFMultiDepthClusterizer_h

// Alpaka includes
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"

#include "DataFormats/ParticleFlowReco/interface/alpaka/PFClusterDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitFractionDeviceCollection.h"

#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringCCLabelsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringVarsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringEdgeVarsDeviceCollection.h"

#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterParams.h"

/**
 * @brief Perform multi-depth PF clusterization on the device.
 *
 * @param queue           Alpaka execution queue.
 * @param outPFCluster    Output PF cluster device collection.
 * @param outPFRecHitFracs Output PF rechit fraction device collection.
 * @param pfCluster       Input PF cluster device collection.
 * @param pfRecHitFracs   Input PF rechit fraction device collection.
 * @param pfRecHit        Input PF rechit device collection.
 * @param params          Pointer to clusterization parameters.
 * @param nClusters       Number of clusters to process.
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
