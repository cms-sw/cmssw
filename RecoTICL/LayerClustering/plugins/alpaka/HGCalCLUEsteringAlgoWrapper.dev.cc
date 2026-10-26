#include <cmath>
#include <cstdint>
#include <span>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "HGCalCLUEsteringAlgoWrapper.h"

#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/core/Clusterer.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  namespace {

    // Copies the seed flag of each rechit from CLUEstering and counts the layer clusters: one per seed, as
    // CLUEstering gives the k-th seed the cluster index k
    struct KernelMarkSeeds {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    const int32_t* isSeed,
                                    HGCalSoARecHitsExtraDeviceCollection::View recHitsClustering,
                                    uint32_t nRecHits) const {
        for (auto i : uniform_elements(acc, nRecHits)) {
          recHitsClustering[i].isSeed() = isSeed[i];
          if (isSeed[i]) {
            alpaka::atomicAdd(acc, &recHitsClustering.numberOfClustersScalar(), 1u);
          }
        }
      }
    };

  }  // namespace

  void HGCalCLUEsteringAlgoWrapper::run(Queue& queue,
                                        uint32_t nRecHits,
                                        float dc,
                                        float kappa,
                                        float outlierDeltaFactor,
                                        bool isScintillator,
                                        std::span<const uint32_t> batchItemSizes,
                                        HGCalSoARecHitsDeviceCollection::ConstView recHits,
                                        HGCalSoARecHitsExtraDeviceCollection::View recHitsClustering) const {
    auto nClusters = make_device_view<unsigned int>(queue, recHitsClustering.numberOfClustersScalar());
    alpaka::memset(queue, nClusters, 0x0);
    if (nRecHits == 0) {
      return;
    }

    clue::ConstPointsDevice<2, float> d_points(queue,
                                               static_cast<int32_t>(nRecHits),
                                               recHits.dim1().data(),
                                               recHits.dim2().data(),
                                               recHits.energy().data(),
                                               recHitsClustering.clusterIndex().data());
    d_points.set_density_uncertainty(recHits.sigmaNoise());
    d_points.set_tags(recHits.detid());

    clue::Clusterer<2> algo(queue, dc, kappa, dc * outlierDeltaFactor);
    if (isScintillator) {
      // Scintillator (BH) cells cluster in (eta, phi)
      constexpr float kTwoPi = 2.f * static_cast<float>(M_PI);
      algo.setWrappedCoordinates(0, 1);
      algo.make_clusters(queue,
                         d_points,
                         batchItemSizes,
                         clue::PeriodicEuclideanMetric<2, float>{0.f, kTwoPi},
                         clue::FlatKernel<float>{0.5f});
    } else {
      algo.make_clusters(queue, d_points, batchItemSizes);
    }

    // The kernel reads is_seed from d_points, which is destroyed when run() returns, possibly before the kernel has
    // run. This is safe: on CUDA CLUEstering frees its device memory in queue order (cudaFreeAsync), on ROCm hipFree
    // waits for the device to finish, and the serial backend runs the kernel immediately.
    constexpr uint32_t blockSize = 256;
    const auto workDivRecHits = make_workdiv<Acc1D>(divide_up_by(nRecHits, blockSize), blockSize);
    alpaka::exec<Acc1D>(
        queue, workDivRecHits, KernelMarkSeeds{}, d_points.view().is_seed().data(), recHitsClustering, nRecHits);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
