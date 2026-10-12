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

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    constexpr bool is_cooperative = std::is_same_v<Device, alpaka::DevCudaRt>;
#else
    constexpr bool is_cooperative = false;
#endif

    const unsigned int wExtend = alpaka::getPreferredWarpSize(alpaka::getDev(queue));

    // For ROCm/CUDA backends, this should be a multiple of warp extent,
    // which can be either 32 or 64.
    // For the CPU backend, threadsPerBlock effectively represents the number
    // of elements per thread and must be set to 1, since some compute kernels
    // rely on thread-private local variables.
    const unsigned int threadsPerBlock_sskernel =
        std::is_same_v<Device, alpaka::DevCpu>
            ? 1
            : (nClusters > 768 ? 256 : ::cms::alpakatools::round_up_by(nClusters, wExtend));

    const unsigned int threadsPerBlock_clkernel =
        std::is_same_v<Device, alpaka::DevCpu>
            ? 1
            : (nClusters > 512 ? 256 : ::cms::alpakatools::round_up_by(nClusters, wExtend));

    const unsigned int threadsPerBlock_prolkernel =
        (nClusters > 512 ? 128 : ::cms::alpakatools::round_up_by(nClusters, wExtend));

    const unsigned int threadsPerBlock_epikernel =
        (nClusters > 512 ? 128 : ::cms::alpakatools::round_up_by(nClusters, wExtend));

    const unsigned int threadsPerBlock_eclcckernel =
        std::is_same_v<Device, alpaka::DevCpu>
            ? 1
            : (nClusters > 768 ? 256 : ::cms::alpakatools::round_up_by(nClusters, wExtend));

    const unsigned int blocks_sskernel = ::cms::alpakatools::divide_up_by(nClusters, threadsPerBlock_sskernel);
    const unsigned int blocks_clkernel = ::cms::alpakatools::divide_up_by(nClusters, threadsPerBlock_clkernel);
    const unsigned int blocks_prolkernel = ::cms::alpakatools::divide_up_by(nClusters, threadsPerBlock_prolkernel);
    const unsigned int blocks_epikernel = ::cms::alpakatools::divide_up_by(nClusters, threadsPerBlock_epikernel);
    const unsigned int blocks_eclcckernel = ::cms::alpakatools::divide_up_by(nClusters, threadsPerBlock_eclcckernel);

    reco::PFMultiDepthClusteringVarsDeviceCollection mdpfClusteringVars{queue, static_cast<int>(nClusters)};
    reco::PFMultiDepthClusteringCCLabelsDeviceCollection mdpfCCLabels{queue, static_cast<int>(nClusters) + 1};
    reco::PFMultiDepthClusteringEdgeVarsDeviceCollection mdpfClusteringEdgeVars{queue, 2 * static_cast<int>(nClusters)};

    alpaka::exec<Acc1D>(queue,
                        ::cms::alpakatools::make_workdiv<Acc1D>(blocks_sskernel, threadsPerBlock_sskernel),
                        ShowerShapeKernel{},
                        mdpfClusteringVars.view(),
                        pfCluster.view(),
                        pfRecHitFracs.view(),
                        pfRecHit.view());
    alpaka::exec<Acc1D>(queue,
                        ::cms::alpakatools::make_workdiv<Acc1D>(blocks_clkernel, threadsPerBlock_clkernel),
                        ConstructLinksKernel{},
                        mdpfCCLabels.view(),
                        mdpfClusteringVars.view(),
                        params);

    const bool do_opt_prologue = enable_multiblock_prologue && (blocks_prolkernel > 1);

    if (do_opt_prologue) {
      // All multi-block prologue kernels must use the same launch parameters.
      reco::PFMultiDepthECLCCPrologueArgsDeviceCollection prologueArgs{queue, static_cast<int>(nClusters)};

      prologueArgs.zeroInitialise(queue);

      alpaka::exec<Acc1D>(queue,
                          ::cms::alpakatools::make_workdiv<Acc1D>(blocks_prolkernel, threadsPerBlock_prolkernel),
                          ECLCCComputeExternNeighsKernel{},
                          prologueArgs.view(),
                          mdpfCCLabels.view());

      if (threadsPerBlock_prolkernel <= 256) {
        constexpr unsigned int max_w_items = 8;
        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks_prolkernel, threadsPerBlock_prolkernel),
                            ECLCCPrologueComputeOffsetsKernel<max_w_items>{},
                            prologueArgs.view(),
                            mdpfCCLabels.view());

        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks_prolkernel, threadsPerBlock_prolkernel),
                            ECLCCFinalizePrologueKernel<max_w_items>{},
                            mdpfClusteringEdgeVars.view(),
                            prologueArgs.view(),
                            mdpfCCLabels.view());
      } else if (threadsPerBlock_prolkernel <= 512) {
        constexpr unsigned int max_w_items = 16;
        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks_prolkernel, threadsPerBlock_prolkernel),
                            ECLCCPrologueComputeOffsetsKernel<max_w_items>{},
                            prologueArgs.view(),
                            mdpfCCLabels.view());

        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks_prolkernel, threadsPerBlock_prolkernel),
                            ECLCCFinalizePrologueKernel<max_w_items>{},
                            mdpfClusteringEdgeVars.view(),
                            prologueArgs.view(),
                            mdpfCCLabels.view());

      } else {
        constexpr unsigned int max_w_items = 32;
        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks_prolkernel, threadsPerBlock_prolkernel),
                            ECLCCPrologueComputeOffsetsKernel<max_w_items>{},
                            prologueArgs.view(),
                            mdpfCCLabels.view());

        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks_prolkernel, threadsPerBlock_prolkernel),
                            ECLCCFinalizePrologueKernel<max_w_items>{},
                            mdpfClusteringEdgeVars.view(),
                            prologueArgs.view(),
                            mdpfCCLabels.view());
      }

      alpaka::exec<Acc1D>(queue,
                          ::cms::alpakatools::make_workdiv<Acc1D>(blocks_prolkernel, threadsPerBlock_prolkernel),
                          ECLCCLoadCrossBlockNeighKernel{},
                          mdpfClusteringEdgeVars.view(),
                          prologueArgs.view(),
                          mdpfCCLabels.view());

    } else {
      if constexpr (std::is_same_v<Device, alpaka::DevCpu>) {
        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(nClusters, 1),
                            ECLCCPrologueNaiveKernel{},
                            mdpfClusteringEdgeVars.view(),
                            mdpfCCLabels.view());
      } else {
        // ECL-CC prologue:
        if (threadsPerBlock_prolkernel <= 256) {
          constexpr unsigned int max_w_items = 8;
          alpaka::exec<Acc1D>(queue,
                              ::cms::alpakatools::make_workdiv<Acc1D>(blocks_prolkernel, threadsPerBlock_prolkernel),
                              ECLCCPrologueKernel<max_w_items>{},
                              mdpfClusteringEdgeVars.view(),
                              mdpfCCLabels.view());
        } else if (threadsPerBlock_prolkernel <= 512) {
          constexpr unsigned int max_w_items = 16;
          alpaka::exec<Acc1D>(queue,
                              ::cms::alpakatools::make_workdiv<Acc1D>(blocks_prolkernel, threadsPerBlock_prolkernel),
                              ECLCCPrologueKernel<max_w_items>{},
                              mdpfClusteringEdgeVars.view(),
                              mdpfCCLabels.view());
        } else {
          constexpr unsigned int max_w_items = 32;
          alpaka::exec<Acc1D>(queue,
                              ::cms::alpakatools::make_workdiv<Acc1D>(blocks_prolkernel, threadsPerBlock_prolkernel),
                              ECLCCPrologueKernel<max_w_items>{},
                              mdpfClusteringEdgeVars.view(),
                              mdpfCCLabels.view());
        }
      }
    }

    // Launch ECL-CC algorithm:
    // ECL-CC init stage:
    alpaka::exec<Acc1D>(queue,
                        ::cms::alpakatools::make_workdiv<Acc1D>(blocks_eclcckernel, threadsPerBlock_eclcckernel),
                        ECLCCInitKernel{},
                        mdpfCCLabels.view(),
                        mdpfClusteringEdgeVars.view());

    // ECL-CC run low-degree hooking:
    alpaka::exec<Acc1D>(queue,
                        ::cms::alpakatools::make_workdiv<Acc1D>(blocks_eclcckernel, threadsPerBlock_eclcckernel),
                        ECLCCLowDegreeComputeKernel{},
                        mdpfCCLabels.view(),
                        mdpfClusteringEdgeVars.view());
    // ECL-CC run mid-degree hooking:
    alpaka::exec<Acc1D>(queue,
                        ::cms::alpakatools::make_workdiv<Acc1D>(blocks_eclcckernel, threadsPerBlock_eclcckernel),
                        ECLCCMidDegreeComputeKernel{},
                        mdpfCCLabels.view(),
                        mdpfClusteringEdgeVars.view());
    // ECL-CC run high-degree hooking:
    alpaka::exec<Acc1D>(queue,
                        ::cms::alpakatools::make_workdiv<Acc1D>(blocks_eclcckernel, threadsPerBlock_eclcckernel),
                        ECLCCHighDegreeComputeKernel{},
                        mdpfCCLabels.view(),
                        mdpfClusteringEdgeVars.view());
    // ECL-CC run finalizing stage:
    alpaka::exec<Acc1D>(queue,
                        ::cms::alpakatools::make_workdiv<Acc1D>(blocks_eclcckernel, threadsPerBlock_eclcckernel),
                        ECLCCFlattenKernel{},
                        mdpfCCLabels.view());

    const bool do_opt_epilogue = enable_multiblock_epilogue && (blocks_epikernel > 1);

    if (do_opt_epilogue) {
      reco::PFMultiDepthECLCCEpilogueArgsDeviceCollection epilogueArgs{queue, static_cast<int>(nClusters)};

      epilogueArgs.zeroInitialise(queue);

      // All multi-block epilogue kernels must use the same launch parameters.

      alpaka::exec<Acc1D>(queue,
                          ::cms::alpakatools::make_workdiv<Acc1D>(blocks_epikernel, threadsPerBlock_epikernel),
                          ECLCCEpilogueRecHitFracOffsetsKernel{},
                          epilogueArgs.view(),
                          mdpfCCLabels.view(),
                          pfCluster.view());
      if (threadsPerBlock_epikernel <= 256) {
        constexpr unsigned int max_w_items = 8;
        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks_epikernel, threadsPerBlock_epikernel),
                            ECLCCEpilogueCCOffsetsKernel<max_w_items>{},
                            outPFCluster.view(),
                            epilogueArgs.view(),
                            mdpfCCLabels.view(),
                            pfCluster.view());

        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks_epikernel, threadsPerBlock_epikernel),
                            ECLCCFinalizeEpilogueKernel<max_w_items, is_cooperative>{},
                            outPFCluster.view(),
                            outPFRecHitFracs.view(),
                            epilogueArgs.view(),
                            mdpfCCLabels.view(),
                            pfCluster.view(),
                            pfRecHitFracs.view(),
                            pfRecHit.view());

      } else if (threadsPerBlock_epikernel <= 256) {
        constexpr unsigned int max_w_items = 16;
        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks_epikernel, threadsPerBlock_epikernel),
                            ECLCCEpilogueCCOffsetsKernel<max_w_items>{},
                            outPFCluster.view(),
                            epilogueArgs.view(),
                            mdpfCCLabels.view(),
                            pfCluster.view());

        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks_epikernel, threadsPerBlock_epikernel),
                            ECLCCFinalizeEpilogueKernel<max_w_items, is_cooperative>{},
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
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks_epikernel, threadsPerBlock_epikernel),
                            ECLCCEpilogueCCOffsetsKernel<max_w_items>{},
                            outPFCluster.view(),
                            epilogueArgs.view(),
                            mdpfCCLabels.view(),
                            pfCluster.view());

        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(blocks_epikernel, threadsPerBlock_epikernel),
                            ECLCCFinalizeEpilogueKernel<max_w_items, is_cooperative>{},
                            outPFCluster.view(),
                            outPFRecHitFracs.view(),
                            epilogueArgs.view(),
                            mdpfCCLabels.view(),
                            pfCluster.view(),
                            pfRecHitFracs.view(),
                            pfRecHit.view());
      }

      alpaka::exec<Acc1D>(queue,
                          ::cms::alpakatools::make_workdiv<Acc1D>(blocks_epikernel, threadsPerBlock_epikernel),
                          ECLCCLoadSeedsKernel{},
                          outPFCluster.view(),
                          epilogueArgs.view());

    } else {
      if constexpr (std::is_same_v<Device, alpaka::DevCpu>) {
        alpaka::exec<Acc1D>(queue,
                            ::cms::alpakatools::make_workdiv<Acc1D>(nClusters, 1),
                            ECLCCEpilogueNaiveKernel{},
                            outPFCluster.view(),
                            outPFRecHitFracs.view(),
                            mdpfCCLabels.view(),
                            pfCluster.view(),
                            pfRecHitFracs.view(),
                            pfRecHit.view());
      } else {
        // ECL-CC epilogue:
        if (threadsPerBlock_epikernel <= 256) {
          constexpr unsigned int max_w_items = 8;
          alpaka::exec<Acc1D>(queue,
                              ::cms::alpakatools::make_workdiv<Acc1D>(blocks_epikernel, threadsPerBlock_epikernel),
                              ECLCCEpilogueKernel<max_w_items, is_cooperative>{},
                              outPFCluster.view(),
                              outPFRecHitFracs.view(),
                              mdpfCCLabels.view(),
                              pfCluster.view(),
                              pfRecHitFracs.view(),
                              pfRecHit.view());

        } else if (threadsPerBlock_epikernel <= 512) {
          constexpr unsigned int max_w_items = 16;
          alpaka::exec<Acc1D>(queue,
                              ::cms::alpakatools::make_workdiv<Acc1D>(blocks_epikernel, threadsPerBlock_epikernel),
                              ECLCCEpilogueKernel<max_w_items, is_cooperative>{},
                              outPFCluster.view(),
                              outPFRecHitFracs.view(),
                              mdpfCCLabels.view(),
                              pfCluster.view(),
                              pfRecHitFracs.view(),
                              pfRecHit.view());
        } else {
          constexpr unsigned int max_w_items = 32;
          alpaka::exec<Acc1D>(queue,
                              ::cms::alpakatools::make_workdiv<Acc1D>(blocks_epikernel, threadsPerBlock_epikernel),
                              ECLCCEpilogueKernel<max_w_items, is_cooperative>{},
                              outPFCluster.view(),
                              outPFRecHitFracs.view(),
                              mdpfCCLabels.view(),
                              pfCluster.view(),
                              pfRecHitFracs.view(),
                              pfRecHit.view());
        }
      }
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::eclcc
