#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalTilesConstants.h"

#include "HGCalLayerClustersSoAAlgoWrapper.h"
#include "ConstantsForClusters.h"

#include "CLUEAlgoAlpaka.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;
  using namespace hgcal::constants;

  // Set energy and number of hits in each clusters
  class HGCalLayerClustersSoAAlgoKernelEnergy {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const unsigned int numer_of_clusters,
                                  const HGCalSoARecHitsDeviceCollection::ConstView input_rechits_soa,
                                  const HGCalSoARecHitsExtraDeviceCollection::ConstView input_clusters_soa,
                                  HGCalSoAClustersDeviceCollection::View outputs) const {
      // make a strided loop over the kernel grid, covering up to "size" elements
      for (int32_t i : uniform_elements(acc, input_rechits_soa.metadata().size())) {
        // Skip unassigned rechits
        if (input_clusters_soa[i].clusterIndex() == kInvalidCluster) {
          continue;
        }
        auto clIdx = input_clusters_soa[i].clusterIndex();
        alpaka::atomicAdd(acc, &outputs[clIdx].energy(), input_rechits_soa[i].weight());
        alpaka::atomicAdd(acc, &outputs[clIdx].cells(), 1);
        if (input_clusters_soa[i].isSeed() == 1) {
          outputs[clIdx].seed() = input_rechits_soa[i].detid();
        }
      }
    }
  };

  // Kernel to find the max for every cluster
  class HGCalLayerClustersSoAAlgoKernelPositionByHits {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const unsigned int numer_of_clusters,
                                  float thresholdW0,
                                  float positionDeltaRho2,
                                  const HGCalSoARecHitsDeviceCollection::ConstView input_rechits_soa,
                                  const HGCalSoARecHitsExtraDeviceCollection::ConstView input_clusters_soa,
                                  HGCalSoAClustersDeviceCollection::View outputs,
                                  HGCalSoAClustersExtraDeviceCollection::View outputs_service) const {
      // make a strided loop over the kernel grid, covering up to "size" elements
      for (int32_t hit_index : uniform_elements(acc, input_rechits_soa.metadata().size())) {
        const int cluster_index = input_clusters_soa[hit_index].clusterIndex();

        // Bail out if you are not part of any cluster
        if (cluster_index == kInvalidCluster) {
          continue;
        }

        alpaka::atomicAdd(acc, &outputs_service[cluster_index].total_weight(), input_rechits_soa[hit_index].weight());
        // Read the current seed index, and the associated energy.
        int clusterSeed = outputs_service[cluster_index].maxEnergyIndex();
        float clusterEnergy = (clusterSeed == kInvalidIndex) ? 0.f : input_rechits_soa[clusterSeed].weight();

        while (input_rechits_soa[hit_index].weight() > clusterEnergy) {
          // If output_service[cluster_index].maxEnergyIndex() did not change,
          // store the new value and exit the loop.  Otherwise return the value
          // that has been updated, and decide again if the maximum needs to be
          // updated.
          int seed = alpaka::atomicCas(acc, &outputs_service[cluster_index].maxEnergyIndex(), clusterSeed, hit_index);
          if (seed == hit_index) {
            // atomicCas has stored the new value.
            break;
          } else {
            // Update the seed index and re-read the associated energy.
            clusterSeed = seed;
            clusterEnergy = (clusterSeed == kInvalidIndex) ? 0.f : input_rechits_soa[clusterSeed].weight();
          }
        }  // CAS
      }  // uniform_elements
    }  // operator()
  };

  // Real Kernel position
  class HGCalLayerClustersSoAAlgoKernelPositionByHits2 {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const unsigned int numer_of_clusters,
                                  float thresholdW0,
                                  float positionDeltaRho2,
                                  const HGCalSoARecHitsDeviceCollection::ConstView input_rechits_soa,
                                  const HGCalSoARecHitsExtraDeviceCollection::ConstView input_clusters_soa,
                                  HGCalSoAClustersDeviceCollection::View outputs,
                                  HGCalSoAClustersExtraDeviceCollection::View outputs_service) const {
      // make a strided loop over the kernel grid, covering up to "size" elements
      for (int32_t hit_index : uniform_elements(acc, input_rechits_soa.metadata().size())) {
        const int cluster_index = input_clusters_soa[hit_index].clusterIndex();

        // Bail out if you are not part of any cluster
        if (cluster_index == kInvalidCluster) {
          continue;
        }
        const int max_energy_index = outputs_service[cluster_index].maxEnergyIndex();

        //for silicon only just use 1+6 cells = 1.3cm for all thicknesses
        const float d1 = input_rechits_soa[hit_index].dim1() - input_rechits_soa[max_energy_index].dim1();
        const float d2 = input_rechits_soa[hit_index].dim2() - input_rechits_soa[max_energy_index].dim2();
        if (std::fmaf(d1, d1, d2 * d2) > positionDeltaRho2) {
          continue;
        }
        float Wi = std::max(thresholdW0 + std::log(input_rechits_soa[hit_index].weight() /
                                                   outputs_service[cluster_index].total_weight()),
                            0.f);
        alpaka::atomicAdd(acc, &outputs[cluster_index].x(), input_rechits_soa[hit_index].dim1() * Wi);
        alpaka::atomicAdd(acc, &outputs[cluster_index].y(), input_rechits_soa[hit_index].dim2() * Wi);
        alpaka::atomicAdd(acc, &outputs_service[cluster_index].total_weight_log(), Wi);
      }  // uniform_elements
    }  // operator()
  };

  // Besides the final position, add also the DetId of the seed of each cluster
  class HGCalLayerClustersSoAAlgoKernelPositionByHits3 {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const unsigned int numer_of_clusters,
                                  float thresholdW0,
                                  float positionDeltaRho2,
                                  const HGCalSoARecHitsDeviceCollection::ConstView input_rechits_soa,
                                  const HGCalSoARecHitsExtraDeviceCollection::ConstView input_clusters_soa,
                                  HGCalSoAClustersDeviceCollection::View outputs,
                                  HGCalSoAClustersExtraDeviceCollection::View outputs_service) const {
      // make a strided loop over the kernel grid, covering up to "size" elements
      for (int32_t cluster_index : uniform_elements(acc, outputs.metadata().size())) {
        const int max_energy_index = outputs_service[cluster_index].maxEnergyIndex();

        if (outputs_service[cluster_index].total_weight_log() > 0.f) {
          float inv_tot_weight = 1.f / outputs_service[cluster_index].total_weight_log();
          outputs[cluster_index].x() *= inv_tot_weight;
          outputs[cluster_index].y() *= inv_tot_weight;
        } else {
          outputs[cluster_index].x() = input_rechits_soa[max_energy_index].dim1();
          outputs[cluster_index].y() = input_rechits_soa[max_energy_index].dim2();
        }
        outputs[cluster_index].z() = input_rechits_soa[max_energy_index].dim3();
      }  // uniform_elements
    }  // operator()
  };

  void HGCalLayerClustersSoAAlgoWrapper::run(Queue& queue,
                                             const unsigned int size,
                                             float thresholdW0,
                                             float positionDeltaRho2,
                                             const HGCalSoARecHitsDeviceCollection::ConstView input_rechits_soa,
                                             const HGCalSoARecHitsExtraDeviceCollection::ConstView input_clusters_soa,
                                             HGCalSoAClustersDeviceCollection::View outputs,
                                             HGCalSoAClustersExtraDeviceCollection::View outputs_service) const {
    auto x = cms::alpakatools::make_device_view<float>(queue, outputs.x(), size);
    alpaka::memset(queue, x, 0x0);
    auto y = cms::alpakatools::make_device_view<float>(queue, outputs.y(), size);
    alpaka::memset(queue, y, 0x0);
    auto z = cms::alpakatools::make_device_view<float>(queue, outputs.z(), size);
    alpaka::memset(queue, z, 0x0);
    auto seed = cms::alpakatools::make_device_view<int>(queue, outputs.seed(), size);
    alpaka::memset(queue, seed, 0x0);
    auto energy = cms::alpakatools::make_device_view<float>(queue, outputs.energy(), size);
    alpaka::memset(queue, energy, 0x0);
    auto cells = cms::alpakatools::make_device_view<int>(queue, outputs.cells(), size);
    alpaka::memset(queue, cells, 0x0);
    auto total_weight = cms::alpakatools::make_device_view<float>(queue, outputs_service.total_weight(), size);
    alpaka::memset(queue, total_weight, 0x0);
    auto total_weight_log = cms::alpakatools::make_device_view<float>(queue, outputs_service.total_weight_log(), size);
    alpaka::memset(queue, total_weight_log, 0x0);
    auto maxEnergyValue = cms::alpakatools::make_device_view<float>(queue, outputs_service.maxEnergyValue(), size);
    alpaka::memset(queue, maxEnergyValue, 0x0);
    auto maxEnergyIndex = cms::alpakatools::make_device_view<int>(queue, outputs_service.maxEnergyIndex(), size);
    alpaka::memset(queue, maxEnergyIndex, kInvalidIndexByte);

    // use 64 items per group (this value is arbitrary, but it's a reasonable starting point)
    uint32_t items = 64;

    // use as many groups as needed to cover the whole problem
    uint32_t groups = divide_up_by(input_rechits_soa.metadata().size(), items);

    // map items to
    //   - threads with a single element per thread on a GPU backend
    //   - elements within a single thread on a CPU backend
    auto workDiv = make_workdiv<Acc1D>(groups, items);

    alpaka::exec<Acc1D>(
        queue, workDiv, HGCalLayerClustersSoAAlgoKernelEnergy{}, size, input_rechits_soa, input_clusters_soa, outputs);
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        HGCalLayerClustersSoAAlgoKernelPositionByHits{},
                        size,
                        thresholdW0,
                        positionDeltaRho2,
                        input_rechits_soa,
                        input_clusters_soa,
                        outputs,
                        outputs_service);
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        HGCalLayerClustersSoAAlgoKernelPositionByHits2{},
                        size,
                        thresholdW0,
                        positionDeltaRho2,
                        input_rechits_soa,
                        input_clusters_soa,
                        outputs,
                        outputs_service);
    uint32_t group_clusters = divide_up_by(size, items);
    auto workDivClusters = make_workdiv<Acc1D>(group_clusters, items);
    alpaka::exec<Acc1D>(queue,
                        workDivClusters,
                        HGCalLayerClustersSoAAlgoKernelPositionByHits3{},
                        size,
                        thresholdW0,
                        positionDeltaRho2,
                        input_rechits_soa,
                        input_clusters_soa,
                        outputs,
                        outputs_service);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
