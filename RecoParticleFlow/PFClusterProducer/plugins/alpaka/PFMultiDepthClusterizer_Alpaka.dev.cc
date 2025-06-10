#include <Eigen/Core>
#include <Eigen/Dense>

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
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitTopologyRecord.h"

#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterizer_Alpaka.h"

#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringVarsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringEdgeVarsDeviceCollection.h"

#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthShowerShape.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthConstructLinks.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthECLCCPrologue.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterECLCC.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthECLCCEpilogue.h"

/**
 * @file PFMultiDepthClusterizer_Alpaka.cc
 * @brief Alpaka-based particle flow multi-depth clustering using Alpaka.
 *
 * This file implements the complete multi-depth clustering pipeline for particle flow reconstruction.
 * It constructs cluster links, builds the adjacency graph, detects connected components using 
 * ECL-CC, and aggregates component-level information.
 *
 * The pipeline stages include:
 * - Link construction between clusters based on geometric criteria.
 * - Adjacency graph creation (compressed sparse row format).
 * - ECL-CC connected components detection (low/mid/high degree partitioned).
 * - Component labeling and rechit energy aggregation.
 * 
 * This code uses warp-scope and block-scope optimizations, dynamic work partitioning,
 * and masked collectives.
 */

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  enum class ECLCCMethod { INIT, LOW, MID, HIGH, FLATTEN, INVALID_METHOD };

  /**
 * @class CCGAlgorithmLauncher
 * @brief Kernel driver for ECL-CC algorithm phases.
 * Launches a specific ECL-CC phase depending on the selected method tag.
 *
 * @tparam TAcc   Alpaka accelerator type.
 * @tparam CCAlgo Connected components algorithm class.
 * @tparam method ECL-CC processing stage (INIT, LOW, MID, HIGH, FLATTEN)
 */

  template <typename TAcc, typename CCAlgo, ECLCCMethod method = ECLCCMethod::INVALID_METHOD>
  class CCGAlgorithmLauncher {
  public:
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        CCAlgo cc_algo,
        reco::PFMultiDepthClusteringVarsDeviceCollection::View mdpfClusteringVars,
        const reco::PFMultiDepthClusteringEdgeVarsDeviceCollection::ConstView mdpfClusteringEdgeVars) const -> void {
      static_assert(method != ECLCCMethod::INVALID_METHOD, "Incorrect method.\n");
      //
      if constexpr (method == ECLCCMethod::INIT) {
        cc_algo.init(acc, mdpfClusteringVars, mdpfClusteringEdgeVars);
      } else if constexpr (method == ECLCCMethod::LOW) {
        cc_algo.compute_low_degree_vertices(acc, mdpfClusteringVars, mdpfClusteringEdgeVars);
      } else if constexpr (method == ECLCCMethod::MID) {
        cc_algo.compute_mid_degree_vertices(acc, mdpfClusteringVars, mdpfClusteringEdgeVars);
      } else if constexpr (method == ECLCCMethod::HIGH) {
        cc_algo.compute_high_degree_vertices(acc, mdpfClusteringVars, mdpfClusteringEdgeVars);
      } else {
        cc_algo.flatten(acc, mdpfClusteringVars, mdpfClusteringEdgeVars);
      }
    }
  };

  /**
 * @brief Executes the complete multi-depth clustering pipeline.
 * The apply() method launches:
 * - ConstructLinksKernel:  Build geometric links between clusters.
 * - ECLCCPrologueKernel:   Build adjacency graph.
 * - CCGAlgorithmLauncher stages (INIT, LOW, MID, HIGH, FLATTEN): ECL-CC component detection.
 * - ECLCCEpilogueKernel: Aggregate rechit energies and assign final component indices.
 * 
 * @param queue Alpaka execution queue.
 * @param outPFCluster Device collection of clusters (output).
 * @param pfRecHitFracs Device collection of PFRecHits fractions (output)
 * @param pfClusters Device collection of clusters variables(input).
 * @param pfRecHitFracs Device collection of PFRecHits fractions (input)
 * @param pfRecHit Device collection of PFRecHits (input).
 *
 */
  void PFMultiDepthClusterizer_Alpaka::apply(Queue& queue,
                                             reco::PFClusterDeviceCollection& outPFCluster,
                                             reco::PFRecHitFractionDeviceCollection& outPFRecHitFracs,
                                             const reco::PFClusterDeviceCollection& pfCluster,
                                             const reco::PFRecHitFractionDeviceCollection& pfRecHitFracs,
                                             const reco::PFRecHitDeviceCollection& pfRecHit,
					     const PFMultiDepthClusterParams* params,
					     const int nClusters) {
    const unsigned int threadsPerBlock = 256;
    const unsigned int blocks = ::cms::alpakatools::divide_up_by(nClusters, threadsPerBlock);
    //
    reco::PFMultiDepthClusteringVarsDeviceCollection mdpfClusteringVars{nClusters + 1, queue};
    reco::PFMultiDepthClusteringEdgeVarsDeviceCollection mdpfClusteringEdgeVars{2 * nClusters, queue};
    //
    alpaka::exec<Acc1D>(queue,
                        ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        ShowerShapeKernel{},
                        mdpfClusteringVars.view(),
                        pfCluster.view(),
                        pfRecHitFracs.view(),
                        pfRecHit.view());
    //
    alpaka::exec<Acc1D>(queue,
                        ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        ConstructLinksKernel{},
                        mdpfClusteringVars.view(),
                        params);

    // ECL-CC prologue:
    if (nClusters < 256) {
      constexpr unsigned int max_w_items = 8;
      alpaka::exec<Acc1D>(queue,
                          ::cms::alpakatools::make_workdiv<Acc1D>(1, nClusters),
                          ECLCCPrologueKernel<max_w_items>{},
                          mdpfClusteringEdgeVars.view(),
                          mdpfClusteringVars.view());
    } else if (nClusters < 512) {
      constexpr unsigned int max_w_items = 16;
      alpaka::exec<Acc1D>(queue,
                          ::cms::alpakatools::make_workdiv<Acc1D>(1, nClusters),
                          ECLCCPrologueKernel<max_w_items>{},
                          mdpfClusteringEdgeVars.view(),
                          mdpfClusteringVars.view());
    } else {
      constexpr unsigned int max_w_items = 32;
      alpaka::exec<Acc1D>(queue,
                          ::cms::alpakatools::make_workdiv<Acc1D>(1, nClusters),
                          ECLCCPrologueKernel<max_w_items>{},
                          mdpfClusteringEdgeVars.view(),
                          mdpfClusteringVars.view());
    }
    // Create and launch ECL-CC algorithm:

    auto workl = ::cms::alpakatools::make_device_buffer<int[]>(queue, nClusters);
    auto tp = ::cms::alpakatools::make_device_buffer<int[]>(queue, 4);
    // Create algorithm internal resources:
    auto cc_args = CCGAlgorithmArgs<decltype(workl)>(queue, workl, tp);
    // Create algorithm
    auto cc_algo = CCGAlgorithm<decltype(cc_args)>(cc_args);
    //
    // ECL-CC init stage:
    alpaka::exec<Acc1D>(queue,
                        ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        CCGAlgorithmLauncher<Acc1D, decltype(cc_algo), ECLCCMethod::INIT>{},
                        cc_algo,
                        mdpfClusteringVars.view(),
                        mdpfClusteringEdgeVars.view());

    // ECL-CC run low-degree hooking:
    alpaka::exec<Acc1D>(queue,
                        ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        CCGAlgorithmLauncher<Acc1D, decltype(cc_algo), ECLCCMethod::LOW>{},
                        cc_algo,
                        mdpfClusteringVars.view(),
                        mdpfClusteringEdgeVars.view());

    // ECL-CC run mid-degree hooking:
    alpaka::exec<Acc1D>(queue,
                        ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        CCGAlgorithmLauncher<Acc1D, decltype(cc_algo), ECLCCMethod::MID>{},
                        cc_algo,
                        mdpfClusteringVars.view(),
                        mdpfClusteringEdgeVars.view());

    // ECL-CC run high-degree hooking:
    alpaka::exec<Acc1D>(queue,
                        ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        CCGAlgorithmLauncher<Acc1D, decltype(cc_algo), ECLCCMethod::HIGH>{},
                        cc_algo,
                        mdpfClusteringVars.view(),
                        mdpfClusteringEdgeVars.view());

    // ECL-CC run finalizing stage:
    alpaka::exec<Acc1D>(queue,
                        ::cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        CCGAlgorithmLauncher<Acc1D, decltype(cc_algo), ECLCCMethod::FLATTEN>{},
                        cc_algo,
                        mdpfClusteringVars.view(),
                        mdpfClusteringEdgeVars.view());

    // ECL-CC epilogue:
    if (nClusters < 256) {
      constexpr unsigned int max_w_items = 8;
      alpaka::exec<Acc1D>(queue,
                          ::cms::alpakatools::make_workdiv<Acc1D>(1, nClusters),
                          ECLCCEpilogueKernel<max_w_items>{},
                          outPFCluster.view(),
                          outPFRecHitFracs.view(),
                          mdpfClusteringVars.view(),
                          pfCluster.view(),
                          pfRecHitFracs.view(),
                          pfRecHit.view());
    } else if (nClusters < 512) {
      constexpr unsigned int max_w_items = 16;
      alpaka::exec<Acc1D>(queue,
                          ::cms::alpakatools::make_workdiv<Acc1D>(1, nClusters),
                          ECLCCEpilogueKernel<max_w_items>{},
                          outPFCluster.view(),
                          outPFRecHitFracs.view(),
                          mdpfClusteringVars.view(),
                          pfCluster.view(),
                          pfRecHitFracs.view(),
                          pfRecHit.view());
    } else {
      constexpr unsigned int max_w_items = 32;
      alpaka::exec<Acc1D>(queue,
                          ::cms::alpakatools::make_workdiv<Acc1D>(1, nClusters),
                          ECLCCEpilogueKernel<max_w_items>{},
                          outPFCluster.view(),
                          outPFRecHitFracs.view(),
                          mdpfClusteringVars.view(),
                          pfCluster.view(),
                          pfRecHitFracs.view(),
                          pfRecHit.view());
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
