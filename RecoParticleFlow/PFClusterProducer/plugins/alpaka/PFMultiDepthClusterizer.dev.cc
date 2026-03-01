#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"

#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterizer.h"

#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringVarsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringCCLabelsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringEdgeVarsDeviceCollection.h"

#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthShowerShape.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthConstructLinks.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthECLCCPrologue.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterECLCC.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthECLCCEpilogue.h"

/**
 * @file PFMultiDepthClusterizer.dev.cc
 * @brief Alpaka-based particle flow multi-depth clustering using Alpaka.
 *
 * This file implements the complete multi-depth clustering pipeline for particle flow reconstruction.
 * It constructs cluster links, builds the adjacency graph, detects connected components using 
 * ECL-CC, and aggregates component-level information.
 *
 * The pipeline stages include:
 * - Shower shape computation
 * - Link construction between clusters based on geometric criteria.
 * - Adjacency graph creation (compressed sparse row format).
 * - ECL-CC connected components detection (low/mid/high degree partitioned).
 * - Component labeling and rechit energy aggregation.
 * 
 * This code uses warp-scope and block-scope optimizations, dynamic work partitioning,
 * and masked collectives.
 */

namespace ALPAKA_ACCELERATOR_NAMESPACE::eclcc {

  /**
 * @brief Executes the complete multi-depth clustering pipeline.
 * 
 * @param queue Alpaka execution queue.
 * @param outPFCluster Device collection of clusters (output).
 * @param pfRecHitFracs Device collection of PFRecHits fractions (output)
 * @param pfCluster Device collection of clusters variables(input).
 * @param pfRecHitFracs Device collection of PFRecHits fractions (input)
 * @param pfRecHit Device collection of PFRecHits (input).
 * @param params Pointer to clusterization parameters. (input).
 * @param nClusters Number of clusters to process.
 */
  void clusterize(Queue& queue,
                  reco::PFClusterDeviceCollection& outPFCluster,
                  reco::PFRecHitFractionDeviceCollection& outPFRecHitFracs,
                  const reco::PFClusterDeviceCollection& pfCluster,
                  const reco::PFRecHitFractionDeviceCollection& pfRecHitFracs,
                  const reco::PFRecHitDeviceCollection& pfRecHit,
                  const PFMultiDepthClusterParams* params,
                  const unsigned int nClusters) {
    const unsigned int wExtend = alpaka::getPreferredWarpSize(alpaka::getDev(queue));
    const unsigned int maxThreadsPerBlock = nClusters <= 768 ? 768 : 256;
    const unsigned int threadsPerBlock = std::min(static_cast<alpaka_common::Idx>(maxThreadsPerBlock),
                                                  ::cms::alpakatools::round_up_by(nClusters, wExtend));

    const unsigned int blocks = ::cms::alpakatools::divide_up_by(nClusters, threadsPerBlock);

    reco::PFMultiDepthClusteringVarsDeviceCollection mdpfClusteringVars{queue, static_cast<int>(nClusters)};
    reco::PFMultiDepthClusteringCCLabelsDeviceCollection mdpfCCLabels{queue, static_cast<int>(nClusters) + 1};
    reco::PFMultiDepthClusteringEdgeVarsDeviceCollection mdpfClusteringEdgeVars{queue, 2 * static_cast<int>(nClusters)};

    alpaka::exec<Acc1D>(queue,
                        ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        ShowerShapeKernel{},
                        mdpfClusteringVars.view(),
                        pfCluster.view(),
                        pfRecHitFracs.view(),
                        pfRecHit.view());
    alpaka::exec<Acc1D>(queue,
                        ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        ConstructLinksKernel{},
                        mdpfCCLabels.view(),
                        mdpfClusteringVars.view(),
                        params);
    // ECL-CC prologue:
    if (threadsPerBlock <= 256) {
      constexpr unsigned int max_w_items = 8;
      if (blocks == 1) {
        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                            ECLCCPrologueKernel<max_w_items>{},
                            mdpfClusteringEdgeVars.view(),
                            mdpfCCLabels.view());
      } else {
        constexpr bool multi_block = true;
        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                            ECLCCPrologueKernel<max_w_items, multi_block>{},
                            mdpfClusteringEdgeVars.view(),
                            mdpfCCLabels.view());
      }
    } else if (threadsPerBlock <= 512) {
      constexpr unsigned int max_w_items = 16;
      alpaka::exec<Acc1D>(queue,
                          ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                          ECLCCPrologueKernel<max_w_items>{},
                          mdpfClusteringEdgeVars.view(),
                          mdpfCCLabels.view());
    } else {
      constexpr unsigned int max_w_items = 32;
      alpaka::exec<Acc1D>(queue,
                          ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                          ECLCCPrologueKernel<max_w_items>{},
                          mdpfClusteringEdgeVars.view(),
                          mdpfCCLabels.view());
    }
    // Launch ECL-CC algorithm:
    // ECL-CC init stage:
    alpaka::exec<Acc1D>(queue,
                        ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        ECLCCInitKernel{},
                        mdpfCCLabels.view(),
                        mdpfClusteringEdgeVars.view());

    // ECL-CC run low-degree hooking:
    alpaka::exec<Acc1D>(queue,
                        ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        ECLCCLowDegreeComputeKernel{},
                        mdpfCCLabels.view(),
                        mdpfClusteringEdgeVars.view());
    // ECL-CC run mid-degree hooking:
    alpaka::exec<Acc1D>(queue,
                        ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        ECLCCMidDegreeComputeKernel{},
                        mdpfCCLabels.view(),
                        mdpfClusteringEdgeVars.view());
    // ECL-CC run high-degree hooking:
    alpaka::exec<Acc1D>(queue,
                        ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        ECLCCHighDegreeComputeKernel{},
                        mdpfCCLabels.view(),
                        mdpfClusteringEdgeVars.view());
    // ECL-CC run finalizing stage:
    alpaka::exec<Acc1D>(queue,
                        ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        ECLCCFlattenKernel{},
                        mdpfCCLabels.view());

    // ECL-CC epilogue:
    if (threadsPerBlock <= 256) {
      constexpr unsigned int max_w_items = 8;
      if (blocks == 1) {
        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                            ECLCCEpilogueKernel<max_w_items>{},
                            outPFCluster.view(),
                            outPFRecHitFracs.view(),
                            mdpfCCLabels.view(),
                            pfCluster.view(),
                            pfRecHitFracs.view(),
                            pfRecHit.view());
      } else {
        constexpr bool multi_block = true;
        constexpr bool cooperative_work = true;
        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                            ECLCCEpilogueKernel<max_w_items, cooperative_work, multi_block>{},
                            outPFCluster.view(),
                            outPFRecHitFracs.view(),
                            mdpfCCLabels.view(),
                            pfCluster.view(),
                            pfRecHitFracs.view(),
                            pfRecHit.view());
      }
    } else if (threadsPerBlock <= 512) {
      constexpr unsigned int max_w_items = 16;
      alpaka::exec<Acc1D>(queue,
                          ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                          ECLCCEpilogueKernel<max_w_items>{},
                          outPFCluster.view(),
                          outPFRecHitFracs.view(),
                          mdpfCCLabels.view(),
                          pfCluster.view(),
                          pfRecHitFracs.view(),
                          pfRecHit.view());
    } else {
      constexpr unsigned int max_w_items = 32;
      alpaka::exec<Acc1D>(queue,
                          ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                          ECLCCEpilogueKernel<max_w_items>{},
                          outPFCluster.view(),
                          outPFRecHitFracs.view(),
                          mdpfCCLabels.view(),
                          pfCluster.view(),
                          pfRecHitFracs.view(),
                          pfRecHit.view());
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::eclcc
