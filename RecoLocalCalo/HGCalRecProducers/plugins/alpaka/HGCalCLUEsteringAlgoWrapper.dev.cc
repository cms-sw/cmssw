#include <cmath>
#include <limits>
#include <cstdint>
#include <span>

#include <alpaka/alpaka.hpp>
#include <xtd/xtd.h>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "HGCalCLUEsteringAlgoWrapper.h"
#include "ConstantsForClusters.h"

#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/core/Clusterer.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"

// rho / delta / nearestHigher are only needed for dumping: no reconstruction
// module reads them.
#define DUMP_CLUSTERS 0

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;
  using namespace hgcal::constants;

  namespace {

    struct KernelSetSeeds {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    const int32_t* seeds,
                                    HGCalSoARecHitsExtraDeviceCollection::View outputs,
                                    uint32_t nseeds) const {
        for (auto k : uniform_elements(acc, nseeds)) {
          outputs[seeds[k]].isSeed() = 1;
        }
      }
    };

    //for dumping
    struct CopyClueIntermediatesKernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    const float* rho,
                                    const int32_t* nearestHigher,
                                    HGCalSoARecHitsDeviceCollection::ConstView inputs,
                                    HGCalSoARecHitsExtraDeviceCollection::View outputs,
                                    const uint32_t size,
                                    const bool isScintillator) const {
        constexpr auto kTwoPi = 2.f * static_cast<float>(M_PI);
        for (auto i : uniform_elements(acc, size)) {
          outputs[i].rho() = rho[i];
          const int32_t nh = nearestHigher[i];
          if (nh < 0) {
            outputs[i].nearestHigher() = kInvalidNearestHigher;
            outputs[i].delta() = std::numeric_limits<float>::max();
          } else {
            outputs[i].nearestHigher() = static_cast<unsigned int>(nh);
            auto d1 = inputs[nh].dim1() - inputs[i].dim1();
            auto d2 = inputs[nh].dim2() - inputs[i].dim2();
            if (isScintillator) {
              if (d2 > kTwoPi / 2.f)
                d2 -= kTwoPi;
              else if (d2 < -kTwoPi / 2.f)
                d2 += kTwoPi;
            }
            outputs[i].delta() = xtd::sqrt(d1 * d1 + d2 * d2);
          }
        }
      }
    };

  }  // namespace

  void HGCalCLUEsteringAlgoWrapper::run(Queue& queue,
                                        const unsigned int size,
                                        const float dc,
                                        const float kappa,
                                        const float outlierDeltaFactor,
                                        const bool isScintillator,
                                        std::span<const uint32_t> batchItemSizes,
                                        const HGCalSoARecHitsDeviceCollection::ConstView inputs,
                                        HGCalSoARecHitsExtraDeviceCollection::View outputs) const {
    // Nothing to do for an empty event: return 0 clusters
    if (size == 0) {
      auto nClusters = make_device_view<unsigned int>(queue, outputs.numberOfClustersScalar());
      alpaka::memset(queue, nClusters, 0x0);
      return;
    }

    const auto items = 256u;

    auto isSeedView = make_device_view(queue, outputs.isSeed().data(), size);
    alpaka::fill(queue, isSeedView, static_cast<uint8_t>(0));

    clue::ConstPointsDevice<2, float> d_points(queue,
                                               static_cast<int32_t>(size),
                                               inputs.dim1().data(),
                                               inputs.dim2().data(),
                                               inputs.energy().data(),
                                               outputs.clusterIndex().data());
    d_points.set_density_uncertainty(inputs.sigmaNoise());
    d_points.set_tags(inputs.detid());

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
    alpaka::wait(queue);

#if DUMP_CLUSTERS
    {
      const auto& pointsView = d_points.view();
      const auto copyWorkDiv = make_workdiv<Acc1D>(divide_up_by(size, items), items);
      alpaka::exec<Acc1D>(queue,
                          copyWorkDiv,
                          CopyClueIntermediatesKernel{},
                          pointsView.m_rho,
                          pointsView.m_nearest_higher,
                          inputs,
                          outputs,
                          size,
                          isScintillator);
    }
#endif

    auto h_nClusters = make_host_buffer<unsigned int>(queue);
    *h_nClusters.data() = static_cast<unsigned int>(d_points.n_clusters());
    auto d_nClusters = make_device_view<unsigned int>(queue, outputs.numberOfClustersScalar());
    alpaka::memcpy(queue, d_nClusters, h_nClusters);

    auto seeds = algo.getSeeds();
    const auto nseeds = static_cast<uint32_t>(seeds.size());
    if (nseeds > 0) {
      const auto seedGroups = divide_up_by(nseeds, items);
      const auto seedWorkDiv = make_workdiv<Acc1D>(seedGroups, items);
      alpaka::exec<Acc1D>(queue, seedWorkDiv, KernelSetSeeds{}, seeds.data(), outputs, nseeds);
    }
    alpaka::wait(queue);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
