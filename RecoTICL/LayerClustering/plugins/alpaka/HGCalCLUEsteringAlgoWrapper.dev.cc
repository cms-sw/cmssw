#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "HGCalCLUEsteringAlgoWrapper.h"

#include "CLUEstering/core/Clusterer.hpp"
#include "CLUEstering/core/ConvolutionalKernel.hpp"
#include "CLUEstering/core/DistanceMetrics.hpp"
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

  struct HGCalCLUEsteringAlgoWrapper::Clusterer {
    clue::Clusterer<2> algo;
  };

  HGCalCLUEsteringAlgoWrapper::HGCalCLUEsteringAlgoWrapper(float dc,
                                                           float kappa,
                                                           float outlierDeltaFactor,
                                                           bool isScintillator)
      : clusterer_{std::make_unique<Clusterer>(Clusterer{clue::Clusterer<2>(dc, kappa, dc * outlierDeltaFactor)})},
        isScintillator_{isScintillator} {
    if (isScintillator_) {
      clusterer_->algo.setWrappedCoordinates(0, 1);
    }
  }

  HGCalCLUEsteringAlgoWrapper::~HGCalCLUEsteringAlgoWrapper() = default;

  void HGCalCLUEsteringAlgoWrapper::setExtremes(std::pair<float, float> dim1, std::pair<float, float> dim2) {
    const std::pair<float, float> extremes[2] = {dim1, dim2};
    clusterer_->algo.setExtremes(extremes);
  }

  void HGCalCLUEsteringAlgoWrapper::run(Queue& queue,
                                        uint32_t nRecHits,
                                        std::span<const uint32_t> batchItemSizes,
                                        HGCalSoARecHitsDeviceCollection::ConstView recHits,
                                        HGCalSoARecHitsExtraDeviceCollection::View recHitsClustering) {
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

    if (isScintillator_) {
      constexpr float kTwoPi = 2.f * static_cast<float>(M_PI);
      clusterer_->algo.make_clusters(queue,
                                     d_points,
                                     batchItemSizes,
                                     clue::PeriodicEuclideanMetric<2, float>{0.f, kTwoPi},
                                     clue::FlatKernel<float>{0.5f});
    } else {
      clusterer_->algo.make_clusters(queue, d_points, batchItemSizes);
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
