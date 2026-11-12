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

  PFMultiDepthClusterizerLaunchParams initLaunchParamsCache() {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    constexpr bool is_cooperative = std::is_same_v<Device, alpaka::DevCudaRt>;
#else
    constexpr bool is_cooperative = false;
#endif
    // get the list of devices on the current platform
    auto const& devices = cms::alpakatools::devices<Platform>();
    if (devices.empty()) {
      std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "exiting...\n";
      exit(EXIT_FAILURE);
    }

    auto const getMaxThreadsPerBlock = [](auto const& devAcc, auto const& kernel, auto&&... args) -> int {
      auto const attributes =
          alpaka::getFunctionAttributes<Acc1D>(devAcc, kernel, std::forward<decltype(args)>(args)...);

      return attributes.maxThreadsPerBlock;
    };

    auto cache = PFMultiDepthClusterizerLaunchParams{devices.size()};

    // run the test on each device
    for (auto const& device : devices) {
      std::int32_t const dev_id = alpaka::getNativeHandle(device);

      const int showerShapeKernelMaxThreadsPerBlock =
          getMaxThreadsPerBlock(device,
                                ShowerShapeKernel{},
                                reco::PFMultiDepthClusteringVarsDeviceCollection::View{},
                                reco::PFClusterDeviceCollection::View{},
                                reco::PFRecHitFractionDeviceCollection::View{},
                                reco::PFRecHitDeviceCollection::View{});

      cache.store(dev_id, ::eclcc::KernelLabels::ShowerShape, showerShapeKernelMaxThreadsPerBlock);

      const int constructLinksKernelMaxThreadsPerBlock =
          getMaxThreadsPerBlock(device,
                                ConstructLinksKernel{},
                                reco::PFMultiDepthClusteringCCLabelsDeviceCollection::View{},
                                reco::PFMultiDepthClusteringVarsDeviceCollection::View{},
                                0.0,
                                0.0);

      cache.store(dev_id, ::eclcc::KernelLabels::ConstructLinks, constructLinksKernelMaxThreadsPerBlock);

      int prologueThreadsPerBlock = getMaxThreadsPerBlock(device,
                                                          ECLCCPrologueKernel<32>{},
                                                          reco::PFMultiDepthClusteringEdgeVarsDeviceCollection::View{},
                                                          reco::PFMultiDepthClusteringCCLabelsDeviceCollection::View{});

      cache.store(dev_id, ::eclcc::KernelLabels::SBlockPrologue, prologueThreadsPerBlock);

      prologueThreadsPerBlock = getMaxThreadsPerBlock(device,
                                                      ECLCCPrologueComputeOffsetsKernel<32>{},
                                                      reco::PFMultiDepthECLCCPrologueArgsDeviceCollection::View{},
                                                      reco::PFMultiDepthClusteringCCLabelsDeviceCollection::View{});

      prologueThreadsPerBlock =
          std::min(prologueThreadsPerBlock,
                   getMaxThreadsPerBlock(device,
                                         ECLCCFinalizePrologueKernel<32>{},
                                         reco::PFMultiDepthClusteringEdgeVarsDeviceCollection::View{},
                                         reco::PFMultiDepthECLCCPrologueArgsDeviceCollection::View{},
                                         reco::PFMultiDepthClusteringCCLabelsDeviceCollection::View{}));

      cache.store(dev_id, ::eclcc::KernelLabels::MBlockPrologue, prologueThreadsPerBlock);

      int eclccThreadsPerBlock = getMaxThreadsPerBlock(device,
                                                       ECLCCLowDegreeComputeKernel{},
                                                       reco::PFMultiDepthClusteringCCLabelsDeviceCollection::View{},
                                                       reco::PFMultiDepthClusteringEdgeVarsDeviceCollection::View{});

      eclccThreadsPerBlock =
          std::min(eclccThreadsPerBlock,
                   getMaxThreadsPerBlock(device,
                                         ECLCCMidDegreeComputeKernel{},
                                         reco::PFMultiDepthClusteringCCLabelsDeviceCollection::View{},
                                         reco::PFMultiDepthClusteringEdgeVarsDeviceCollection::View{}));

      eclccThreadsPerBlock =
          std::min(eclccThreadsPerBlock,
                   getMaxThreadsPerBlock(device,
                                         ECLCCHighDegreeComputeKernel{},
                                         reco::PFMultiDepthClusteringCCLabelsDeviceCollection::View{},
                                         reco::PFMultiDepthClusteringEdgeVarsDeviceCollection::View{}));

      cache.store(dev_id, ::eclcc::KernelLabels::ECLCC, eclccThreadsPerBlock);

      int epilogueThreadsPerBlock = getMaxThreadsPerBlock(device,
                                                          ECLCCEpilogueKernel<32, is_cooperative>{},
                                                          reco::PFClusterDeviceCollection::View{},
                                                          reco::PFRecHitFractionDeviceCollection::View{},
                                                          reco::PFMultiDepthClusteringCCLabelsDeviceCollection::View{},
                                                          reco::PFClusterDeviceCollection::View{},
                                                          reco::PFRecHitFractionDeviceCollection::View{},
                                                          reco::PFRecHitDeviceCollection::View{});

      cache.store(dev_id, ::eclcc::KernelLabels::SBlockEpilogue, epilogueThreadsPerBlock);

      epilogueThreadsPerBlock = getMaxThreadsPerBlock(device,
                                                      ECLCCEpilogueCCOffsetsKernel<32>{},
                                                      reco::PFClusterDeviceCollection::View{},
                                                      reco::PFMultiDepthECLCCEpilogueArgsDeviceCollection::View{},
                                                      reco::PFMultiDepthClusteringCCLabelsDeviceCollection::View{},
                                                      reco::PFClusterDeviceCollection::View{});

      epilogueThreadsPerBlock =
          std::min(epilogueThreadsPerBlock,
                   getMaxThreadsPerBlock(device,
                                         ECLCCFinalizeEpilogueKernel<32, is_cooperative>{},
                                         reco::PFClusterDeviceCollection::View{},
                                         reco::PFRecHitFractionDeviceCollection::View{},
                                         reco::PFMultiDepthECLCCEpilogueArgsDeviceCollection::View{},
                                         reco::PFMultiDepthClusteringCCLabelsDeviceCollection::View{},
                                         reco::PFClusterDeviceCollection::View{},
                                         reco::PFRecHitFractionDeviceCollection::View{},
                                         reco::PFRecHitDeviceCollection::View{}));

      cache.store(dev_id, ::eclcc::KernelLabels::MBlockEpilogue, prologueThreadsPerBlock);
    }

    return cache;
  }

  template <auto... M, typename F>
  void dispatchKernel(unsigned int const threadsPerBlock, F&& f) {
    if (threadsPerBlock <= 256) {
      constexpr unsigned int max_w_items = 8;
      std::forward<F>(f).template operator()<max_w_items, M...>();
    } else if (threadsPerBlock <= 512) {
      constexpr unsigned int max_w_items = 16;
      std::forward<F>(f).template operator()<max_w_items, M...>();
    } else {
      constexpr unsigned int max_w_items = 32;
      std::forward<F>(f).template operator()<max_w_items, M...>();
    }
  }

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
                  const PFMultiDepthClusterParams& params,
                  const PFMultiDepthClusterizerLaunchParams& launch_params,
                  const unsigned int nClusters) {
    constexpr bool enable_multiblock_prologue = !std::is_same_v<Device, alpaka::DevCpu>;
    constexpr bool enable_multiblock_epilogue = !std::is_same_v<Device, alpaka::DevCpu>;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    constexpr bool is_cooperative = std::is_same_v<Device, alpaka::DevCudaRt>;
#else
    constexpr bool is_cooperative = false;
#endif

    reco::PFMultiDepthClusteringVarsDeviceCollection mdpfClusteringVars{queue, static_cast<int>(nClusters)};
    reco::PFMultiDepthClusteringCCLabelsDeviceCollection mdpfCCLabels{queue, static_cast<int>(nClusters) + 1};
    reco::PFMultiDepthClusteringEdgeVarsDeviceCollection mdpfClusteringEdgeVars{queue, 2 * static_cast<int>(nClusters)};

    const auto devAcc = alpaka::getDev(queue);

    std::int32_t const devId = alpaka::getNativeHandle(devAcc);

    const unsigned int wExtend = alpaka::getPreferredWarpSize(devAcc);

    const double nSigmaEtaSq = params.nSigmaEtaSq;
    const double nSigmaPhiSq = params.nSigmaPhiSq;

    // For ROCm/CUDA backends, this should be a multiple of warp extent,
    // which can be either 32 or 64.
    // For the CPU backend, the default  threads per block effectively represents the number
    // of elements per thread and must be set to 1, since some compute kernels
    // rely on thread-private local variables.
    const unsigned int defaultThreadsPerBlock = ::cms::alpakatools::round_up_by(
        (std::is_same_v<Device, alpaka::DevCpu> ? 1 : (nClusters > 768 ? 256 : nClusters)), wExtend);

    const unsigned int showerShapeKernelThreadsPerBlock =
        launch_params.load(devId, ::eclcc::KernelLabels::ShowerShape, defaultThreadsPerBlock);
    const unsigned int showerShapeKernelBlocks =
        ::cms::alpakatools::divide_up_by(nClusters, showerShapeKernelThreadsPerBlock);

    alpaka::exec<Acc1D>(
        queue,
        ::cms::alpakatools::make_workdiv<Acc1D>(showerShapeKernelBlocks, showerShapeKernelThreadsPerBlock),
        ShowerShapeKernel{},
        mdpfClusteringVars.view(),
        pfCluster.view(),
        pfRecHitFracs.view(),
        pfRecHit.view());

    const unsigned int constructLinksKernelThreadsPerBlock =
        launch_params.load(devId, ::eclcc::KernelLabels::ConstructLinks, defaultThreadsPerBlock);
    const unsigned int constructLinksKernelBlocks =
        ::cms::alpakatools::divide_up_by(nClusters, constructLinksKernelThreadsPerBlock);

    alpaka::exec<Acc1D>(
        queue,
        ::cms::alpakatools::make_workdiv<Acc1D>(constructLinksKernelBlocks, constructLinksKernelThreadsPerBlock),
        ConstructLinksKernel{},
        mdpfCCLabels.view(),
        mdpfClusteringVars.view(),
        nSigmaEtaSq,
        nSigmaPhiSq);

    const bool do_opt_prologue = enable_multiblock_prologue && (nClusters > 512);

    if (do_opt_prologue) {
      // All multi-block prologue kernels must use the same launch parameters.
      reco::PFMultiDepthECLCCPrologueArgsDeviceCollection prologueArgs{queue, static_cast<int>(nClusters)};

      prologueArgs.zeroInitialise(queue);

      const unsigned int prologueKernelThreadsPerBlock =
          launch_params.load(devId, ::eclcc::KernelLabels::MBlockPrologue, defaultThreadsPerBlock);
      const unsigned int prologueKernelBlocks =
          ::cms::alpakatools::divide_up_by(nClusters, prologueKernelThreadsPerBlock);

      const auto& prologueKernelWorkDiv =
          ::cms::alpakatools::make_workdiv<Acc1D>(prologueKernelBlocks, prologueKernelThreadsPerBlock);

      alpaka::exec<Acc1D>(
          queue, prologueKernelWorkDiv, ECLCCComputeExternNeighsKernel{}, prologueArgs.view(), mdpfCCLabels.view());

      dispatchKernel(prologueKernelThreadsPerBlock, [&]<unsigned int N>() {
        alpaka::exec<Acc1D>(queue,
                            prologueKernelWorkDiv,
                            ECLCCPrologueComputeOffsetsKernel<N>{},
                            prologueArgs.view(),
                            mdpfCCLabels.view());
        alpaka::exec<Acc1D>(queue,
                            prologueKernelWorkDiv,
                            ECLCCFinalizePrologueKernel<N>{},
                            mdpfClusteringEdgeVars.view(),
                            prologueArgs.view(),
                            mdpfCCLabels.view());
      });

      alpaka::exec<Acc1D>(queue,
                          prologueKernelWorkDiv,
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
        const unsigned int prologueKernelThreadsPerBlock =
            launch_params.load(devId, ::eclcc::KernelLabels::SBlockPrologue, defaultThreadsPerBlock);
        const unsigned int prologueKernelBlocks =
            ::cms::alpakatools::divide_up_by(nClusters, prologueKernelThreadsPerBlock);
        // ECL-CC prologue:
        dispatchKernel(prologueKernelThreadsPerBlock, [&]<unsigned int N>() {
          alpaka::exec<Acc1D>(
              queue,
              ::cms::alpakatools::make_workdiv<Acc1D>(prologueKernelBlocks, prologueKernelThreadsPerBlock),
              ECLCCPrologueKernel<N>{},
              mdpfClusteringEdgeVars.view(),
              mdpfCCLabels.view());
        });
      }
    }

    const unsigned int eclccKernelThreadsPerBlock =
        launch_params.load(devId, ::eclcc::KernelLabels::ECLCC, defaultThreadsPerBlock);
    const unsigned int eclccKernelBlocks = ::cms::alpakatools::divide_up_by(nClusters, eclccKernelThreadsPerBlock);

    const auto& eclccKernelWorkDiv =
        ::cms::alpakatools::make_workdiv<Acc1D>(eclccKernelBlocks, eclccKernelThreadsPerBlock);

    // Launch ECL-CC algorithm:
    // ECL-CC init stage:
    alpaka::exec<Acc1D>(
        queue, eclccKernelWorkDiv, ECLCCInitKernel{}, mdpfCCLabels.view(), mdpfClusteringEdgeVars.view());

    // ECL-CC run low-degree hooking:
    alpaka::exec<Acc1D>(
        queue, eclccKernelWorkDiv, ECLCCLowDegreeComputeKernel{}, mdpfCCLabels.view(), mdpfClusteringEdgeVars.view());
    // ECL-CC run mid-degree hooking:
    alpaka::exec<Acc1D>(
        queue, eclccKernelWorkDiv, ECLCCMidDegreeComputeKernel{}, mdpfCCLabels.view(), mdpfClusteringEdgeVars.view());
    // ECL-CC run high-degree hooking:
    alpaka::exec<Acc1D>(
        queue, eclccKernelWorkDiv, ECLCCHighDegreeComputeKernel{}, mdpfCCLabels.view(), mdpfClusteringEdgeVars.view());
    // ECL-CC run finalizing stage:
    alpaka::exec<Acc1D>(queue, eclccKernelWorkDiv, ECLCCFlattenKernel{}, mdpfCCLabels.view());

    const bool do_opt_epilogue = enable_multiblock_epilogue && (nClusters > 512);

    if (do_opt_epilogue) {
      reco::PFMultiDepthECLCCEpilogueArgsDeviceCollection epilogueArgs{queue, static_cast<int>(nClusters)};

      epilogueArgs.zeroInitialise(queue);

      const unsigned int epilogueKernelThreadsPerBlock =
          launch_params.load(devId, ::eclcc::KernelLabels::MBlockEpilogue, defaultThreadsPerBlock);
      const unsigned int epilogueKernelBlocks =
          ::cms::alpakatools::divide_up_by(nClusters, epilogueKernelThreadsPerBlock);

      const auto& epilogueKernelWorkDiv =
          ::cms::alpakatools::make_workdiv<Acc1D>(epilogueKernelBlocks, epilogueKernelThreadsPerBlock);

      // All multi-block epilogue kernels must use the same launch parameters.

      alpaka::exec<Acc1D>(queue,
                          epilogueKernelWorkDiv,
                          ECLCCEpilogueRecHitFracOffsetsKernel{},
                          epilogueArgs.view(),
                          mdpfCCLabels.view(),
                          pfCluster.view());

      dispatchKernel<is_cooperative>(epilogueKernelThreadsPerBlock, [&]<unsigned int N, bool coop>() {
        alpaka::exec<Acc1D>(queue,
                            epilogueKernelWorkDiv,
                            ECLCCEpilogueCCOffsetsKernel<N>{},
                            outPFCluster.view(),
                            epilogueArgs.view(),
                            mdpfCCLabels.view(),
                            pfCluster.view());

        alpaka::exec<Acc1D>(queue,
                            epilogueKernelWorkDiv,
                            ECLCCFinalizeEpilogueKernel<N, coop>{},
                            outPFCluster.view(),
                            outPFRecHitFracs.view(),
                            epilogueArgs.view(),
                            mdpfCCLabels.view(),
                            pfCluster.view(),
                            pfRecHitFracs.view(),
                            pfRecHit.view());
      });

      alpaka::exec<Acc1D>(
          queue, epilogueKernelWorkDiv, ECLCCLoadSeedsKernel{}, outPFCluster.view(), epilogueArgs.view());

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
        const unsigned int epilogueKernelThreadsPerBlock =
            launch_params.load(devId, ::eclcc::KernelLabels::SBlockEpilogue, defaultThreadsPerBlock);
        const unsigned int epilogueKernelBlocks =
            ::cms::alpakatools::divide_up_by(nClusters, epilogueKernelThreadsPerBlock);

        dispatchKernel<is_cooperative>(epilogueKernelThreadsPerBlock, [&]<unsigned int N, bool coop>() {
          alpaka::exec<Acc1D>(
              queue,
              ::cms::alpakatools::make_workdiv<Acc1D>(epilogueKernelBlocks, epilogueKernelThreadsPerBlock),
              ECLCCEpilogueKernel<N, coop>{},
              outPFCluster.view(),
              outPFRecHitFracs.view(),
              mdpfCCLabels.view(),
              pfCluster.view(),
              pfRecHitFracs.view(),
              pfRecHit.view());
        });
      }
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::eclcc
