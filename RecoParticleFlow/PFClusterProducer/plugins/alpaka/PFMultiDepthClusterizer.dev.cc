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

#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthECLCCPrologueArgsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthECLCCEpilogueArgsDeviceCollection.h"

#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthECLCCPrologueMultiBlock.h"

#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthECLCCInitEpilogueArgs.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthECLCCEpilogueMultiBlock.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthECLCCFinalizeEpilogue.h"

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
    constexpr bool enable_multiblock_prologue = !std::is_same_v<Device, alpaka::DevCpu>;
    constexpr bool enable_multiblock_epilogue = !std::is_same_v<Device, alpaka::DevCpu>;

    const unsigned int wExtend = alpaka::getPreferredWarpSize(alpaka::getDev(queue));
    const unsigned int maxThreadsPerBlock = std::is_same_v<Device, alpaka::DevCpu> ? 1 : (nClusters <= 768 ? 768 : 256);
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

    const bool do_opt_prologue = enable_multiblock_prologue && (blocks > 1);

    if (do_opt_prologue) {
      reco::PFMultiDepthECLCCPrologueArgsDeviceCollection prologueArgs{queue, static_cast<int>(nClusters)};

      prologueArgs.zeroInitialise(queue);

      alpaka::exec<Acc1D>(queue,
                          ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                          ECLCCComputeExternNeighsKernel{},
                          prologueArgs.view(),
                          mdpfCCLabels.view());

      if (threadsPerBlock <= 256) {
        constexpr unsigned int max_w_items = 8;
        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                            ECLCCPrologueComputeOffsetsKernel<max_w_items>{},
                            prologueArgs.view(),
                            mdpfCCLabels.view());

        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                            ECLCCFinalizePrologueKernel<max_w_items>{},
                            mdpfClusteringEdgeVars.view(),
                            prologueArgs.view(),
                            mdpfCCLabels.view());
      } else if (threadsPerBlock <= 512) {
        constexpr unsigned int max_w_items = 16;
        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                            ECLCCPrologueComputeOffsetsKernel<max_w_items>{},
                            prologueArgs.view(),
                            mdpfCCLabels.view());

        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                            ECLCCFinalizePrologueKernel<max_w_items>{},
                            mdpfClusteringEdgeVars.view(),
                            prologueArgs.view(),
                            mdpfCCLabels.view());

      } else {
        constexpr unsigned int max_w_items = 32;
        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                            ECLCCPrologueComputeOffsetsKernel<max_w_items>{},
                            prologueArgs.view(),
                            mdpfCCLabels.view());

        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                            ECLCCFinalizePrologueKernel<max_w_items>{},
                            mdpfClusteringEdgeVars.view(),
                            prologueArgs.view(),
                            mdpfCCLabels.view());
      }

      alpaka::exec<Acc1D>(queue,
                          ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                          ECLCCLoadCrossBlockNeighKernel{},
                          mdpfClusteringEdgeVars.view(),
                          prologueArgs.view(),
                          mdpfCCLabels.view());

    } else {
      // ECL-CC prologue:
      if (threadsPerBlock <= 256) {
        constexpr unsigned int max_w_items = 8;
        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                            ECLCCPrologueKernel<max_w_items>{},
                            mdpfClusteringEdgeVars.view(),
                            mdpfCCLabels.view());
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

    const bool do_opt_epilogue = enable_multiblock_epilogue && (blocks > 1);

    if (do_opt_epilogue) {
      reco::PFMultiDepthECLCCEpilogueArgsDeviceCollection epilogueArgs{queue, static_cast<int>(nClusters)};

      alpaka::exec<Acc1D>(queue,
                          ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                          ECLCCInitEpilogueArgsKernel{},
                          epilogueArgs.view(),
                          mdpfCCLabels.view());

      alpaka::exec<Acc1D>(queue,
                          ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                          ECLCCEpilogueRecHitFracOffsetsKernel{},
                          epilogueArgs.view(),
                          mdpfCCLabels.view(),
                          pfCluster.view());
      if (threadsPerBlock <= 256) {
        constexpr unsigned int max_w_items = 8;
        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                            ECLCCEpilogueCCOffsetsKernel<max_w_items>{},
                            outPFCluster.view(),
                            epilogueArgs.view(),
                            mdpfCCLabels.view(),
                            pfCluster.view());

        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                            ECLCCFinalizeEpilogueKernel<max_w_items>{},
                            outPFCluster.view(),
                            outPFRecHitFracs.view(),
                            epilogueArgs.view(),
                            mdpfCCLabels.view(),
                            pfCluster.view(),
                            pfRecHitFracs.view(),
                            pfRecHit.view());

      } else if (threadsPerBlock <= 256) {
        constexpr unsigned int max_w_items = 16;
        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                            ECLCCEpilogueCCOffsetsKernel<max_w_items>{},
                            outPFCluster.view(),
                            epilogueArgs.view(),
                            mdpfCCLabels.view(),
                            pfCluster.view());

        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                            ECLCCFinalizeEpilogueKernel<max_w_items>{},
                            outPFCluster.view(),
                            outPFRecHitFracs.view(),
                            epilogueArgs.view(),
                            mdpfCCLabels.view(),
                            pfCluster.view(),
                            pfRecHitFracs.view(),
                            pfRecHit.view());
      } else {
        constexpr unsigned int max_w_items = 32;
        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                            ECLCCEpilogueCCOffsetsKernel<max_w_items>{},
                            outPFCluster.view(),
                            epilogueArgs.view(),
                            mdpfCCLabels.view(),
                            pfCluster.view());

        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                            ECLCCFinalizeEpilogueKernel<max_w_items>{},
                            outPFCluster.view(),
                            outPFRecHitFracs.view(),
                            epilogueArgs.view(),
                            mdpfCCLabels.view(),
                            pfCluster.view(),
                            pfRecHitFracs.view(),
                            pfRecHit.view());
      }

      alpaka::exec<Acc1D>(queue,
                          ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                          ECLCCLoadSeedsKernel{},
                          outPFCluster.view(),
                          epilogueArgs.view());

    } else {
      constexpr bool cooperative_work = true;
      // ECL-CC epilogue:
      if (threadsPerBlock <= 256) {
        constexpr unsigned int max_w_items = 8;
        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                            ECLCCEpilogueKernel<max_w_items, cooperative_work>{},
                            outPFCluster.view(),
                            outPFRecHitFracs.view(),
                            mdpfCCLabels.view(),
                            pfCluster.view(),
                            pfRecHitFracs.view(),
                            pfRecHit.view());

      } else if (threadsPerBlock <= 512) {
        constexpr unsigned int max_w_items = 16;
        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                            ECLCCEpilogueKernel<max_w_items, cooperative_work>{},
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
                            ECLCCEpilogueKernel<max_w_items, cooperative_work>{},
                            outPFCluster.view(),
                            outPFRecHitFracs.view(),
                            mdpfCCLabels.view(),
                            pfCluster.view(),
                            pfRecHitFracs.view(),
                            pfRecHit.view());
      }
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::eclcc
