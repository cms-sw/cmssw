#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "HGCalLayerClustersSoAAlgoWrapper.h"
#include "ConstantsForClusters.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;
  using namespace hgcal::constants;

  namespace {

    // Scintillator (BH) rechits store (eta, phi) in (dim1, dim2). Recover the
    // cartesian transverse position from them and the cell z.
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void etaPhiZToXY(
        const float eta, const float phi, const float z, float& x, float& y) {
      const float r = z / std::sinh(eta);
      x = r * std::cos(phi);
      y = r * std::sin(phi);
    }

    // The most energetic rechit of a layer cluster is found with one atomicMax on this key. The bits of a non-negative
    // float, read as an integer, are ordered as it's value is. For equal energies, ~hit makes the
    // lowest rechit index win. This assumes energy is always positive
    ALPAKA_FN_ACC ALPAKA_FN_INLINE uint64_t energyKey(const float energy, const uint32_t hit) {
      return (static_cast<uint64_t>(std::bit_cast<uint32_t>(energy)) << 32) | ~hit;
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t hitOfEnergyKey(const uint64_t key) { return ~static_cast<uint32_t>(key); }

    // Rechits without a time have a negative time error, and scintillator layer clusters have no time
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool hasTime(const bool isScintillator, const float timeError) {
      return not(isScintillator or timeError < 0.f);
    }

    // The time of a layer cluster and its error, in ns
    struct LayerClusterTime {
      float time;
      float error;
    };

    // The time of a layer cluster without enough rechits with a time
    constexpr LayerClusterTime kNoTime{-99.f, -1.f};

    // The time of a layer cluster from the n times t of its rechits, with weights w = 1 / timeError^2: the weighted
    // mean of the times in the window of about deltaT holding the most of them, widened by half its width on each
    // side. Same algorithm and floating point operations as ComputeClusterTime::fixSizeHighestDensity with its
    // default parameters. t and w are sorted in place, by time.
    ALPAKA_FN_ACC ALPAKA_FN_INLINE LayerClusterTime fixSizeHighestDensity(float* t,
                                                                          float* w,
                                                                          const uint32_t n,
                                                                          const uint32_t minNHits) {
      constexpr float deltaT = 0.210f;  // width of the window, in ns
      constexpr float timeWidthBy = 0.5f;
      if (n < minNHits or n == 0) {
        return kNoTime;
      }

      // insertion sort by increasing time: the fastest for so few elements (a CE-E layer cluster has 3.4 rechits on
      // average)
      for (uint32_t i = 1; i < n; ++i) {
        const float ti = t[i];
        const float wi = w[i];
        uint32_t j = i;
        for (; j > 0 and t[j - 1] > ti; --j) {
          t[j] = t[j - 1];
          w[j] = w[j - 1];
        }
        t[j] = ti;
        w[j] = wi;
      }

      // The window [start, last] starting at each time t[start] holds the times up to t[start] + deltaT + tolerance.
      // Keep the one with the most times; when its width is within tolerance of deltaT, the tolerance shrinks to
      // that difference. As t is sorted, last only moves forward with start, unless the tolerance has just shrunk.
      uint32_t maxElements = 0;
      uint32_t startEl = 0;
      uint32_t endEl = 0;
      float timeW = 0.f;
      float tolerance = 0.05f;
      uint32_t last = 0;
      for (uint32_t start = 0; start < n; ++start) {
        last = std::max(last, start);
        while (last > start and t[last] - t[start] > deltaT + tolerance) {
          --last;
        }
        while (last + 1 < n and t[last + 1] - t[start] <= deltaT + tolerance) {
          ++last;
        }
        if (last - start + 1 > maxElements) {
          maxElements = last - start + 1;
          const float width = t[last] - t[start];
          if (std::abs(deltaT - width) < tolerance) {
            tolerance = std::abs(deltaT - width);
          }
          startEl = start;
          endEl = last;
          timeW = width;
        }
      }

      // the weighted mean of the times within half of the window width around it
      const float halfTimeDiff = timeW * timeWidthBy;
      float sum = 0.f;
      float num = 0.f;
      for (uint32_t ij = 0; ij <= startEl; ++ij) {
        if (t[ij] > t[startEl] - halfTimeDiff) {
          for (uint32_t kl = ij; kl < n and t[kl] < t[endEl] + halfTimeDiff; ++kl) {
            sum += t[kl] * w[kl];
            num += w[kl];
          }
          break;
        }
      }
      if (num == 0.f) {
        return kNoTime;
      }
      // the error is computed in double precision, like in ComputeClusterTime
      return {sum / num, static_cast<float>(1. / std::sqrt(static_cast<double>(num)))};
    }

    // Per rechit -> per layer cluster: the sum of the energies, the number of cells, the DetId of the seed, the key of
    // the most energetic rechit and the number of rechits with a time
    struct KernelLayerClusterEnergy {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    bool isScintillator,
                                    HGCalSoARecHitsDeviceCollection::ConstView recHits,
                                    HGCalSoARecHitsExtraDeviceCollection::ConstView recHitsClustering,
                                    reco::CaloClusterDeviceCollection::View layerClusters,
                                    uint64_t* maxEnergyKey,
                                    uint32_t* timedHits) const {
        for (auto i : uniform_elements(acc, recHits.metadata().size())) {
          const int c = recHitsClustering[i].clusterIndex();
          if (c == kInvalidCluster) {
            continue;
          }
          alpaka::atomicAdd(acc, &layerClusters.energy()[c].energy(), recHits[i].energy());
          alpaka::atomicAdd(acc, &layerClusters.position()[c].cells(), 1);
          alpaka::atomicMax(acc, &maxEnergyKey[c], energyKey(recHits[i].energy(), i));
          if (recHitsClustering[i].isSeed() == 1) {
            layerClusters.indexes()[c].seedID() = recHits[i].detid();
          }
          if (hasTime(isScintillator, recHits[i].timeError())) {
            alpaka::atomicAdd(acc, &timedHits[c], 1u);
          }
        }
      }
    };

    // Per rechit -> per layer cluster: the weighted sums of the transverse positions and the sum of the weights.
    // The scintillator rechits are weighted by their energy, the silicon rechits within 1.3 cm (1+6 cells, for all
    // thicknesses) of the most energetic one by max(thresholdW0 + log(energy / cluster energy), 0).
    // Also copies the time and the time weight (1 / timeError^2) of each rechit with a time to the segment of its layer
    // cluster in times and weights, which starts at timedOffsets[c].
    struct KernelLayerClusterPositionSums {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    float thresholdW0,
                                    float positionDeltaRho2,
                                    bool isScintillator,
                                    HGCalSoARecHitsDeviceCollection::ConstView recHits,
                                    HGCalSoARecHitsExtraDeviceCollection::ConstView recHitsClustering,
                                    reco::CaloClusterDeviceCollection::View layerClusters,
                                    float* positionWeightSum,
                                    const uint64_t* maxEnergyKey,
                                    const uint32_t* timedOffsets,
                                    uint32_t* timedFilled,
                                    float* times,
                                    float* weights) const {
        for (auto i : uniform_elements(acc, recHits.metadata().size())) {
          const int c = recHitsClustering[i].clusterIndex();
          if (c == kInvalidCluster) {
            continue;
          }

          // every rechit with a time takes part in the time, also the ones outside of the position window below
          const float timeError = recHits[i].timeError();
          if (hasTime(isScintillator, timeError)) {
            const uint32_t slot = timedOffsets[c] + alpaka::atomicAdd(acc, &timedFilled[c], 1u);
            times[slot] = recHits[i].time();
            weights[slot] = 1.f / (timeError * timeError);
          }

          if (isScintillator) {
            const float energy = recHits[i].energy();
            float x;
            float y;
            etaPhiZToXY(recHits[i].dim1(), recHits[i].dim2(), recHits[i].dim3(), x, y);
            alpaka::atomicAdd(acc, &layerClusters.position()[c].x(), x * energy);
            alpaka::atomicAdd(acc, &layerClusters.position()[c].y(), y * energy);
            alpaka::atomicAdd(acc, &positionWeightSum[c], energy);
            continue;
          }

          const uint32_t maxHit = hitOfEnergyKey(maxEnergyKey[c]);
          const float d1 = recHits[i].dim1() - recHits[maxHit].dim1();
          const float d2 = recHits[i].dim2() - recHits[maxHit].dim2();
          if (std::fmaf(d1, d1, d2 * d2) > positionDeltaRho2) {
            continue;
          }
          const float weight =
              std::max(thresholdW0 + std::log(recHits[i].energy() / layerClusters.energy()[c].energy()), 0.f);
          alpaka::atomicAdd(acc, &layerClusters.position()[c].x(), recHits[i].dim1() * weight);
          alpaka::atomicAdd(acc, &layerClusters.position()[c].y(), recHits[i].dim2() * weight);
          alpaka::atomicAdd(acc, &positionWeightSum[c], weight);
        }
      }
    };

    // Per layer cluster: the transverse position from the weighted sums, or the one of its most energetic rechit if
    // no rechit has a weight, z and the layer of its most energetic rechit, and the time from the times of its rechits.
    // Also sets the columns not computed yet to the defaults of reco::CaloCluster.
    struct KernelLayerClusterPositionAndTime {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    bool isScintillator,
                                    uint32_t nHitsTime,
                                    HGCalSoARecHitsDeviceCollection::ConstView recHits,
                                    reco::CaloClusterDeviceCollection::View layerClusters,
                                    const float* positionWeightSum,
                                    const uint64_t* maxEnergyKey,
                                    const uint32_t* timedOffsets,
                                    float* times,
                                    float* weights) const {
        for (auto c : uniform_elements(acc, layerClusters.position().metadata().size())) {
          const uint32_t maxHit = hitOfEnergyKey(maxEnergyKey[c]);
          auto position = layerClusters.position()[c];
          if (positionWeightSum[c] > 0.f) {
            const float invWeightSum = 1.f / positionWeightSum[c];
            position.x() *= invWeightSum;
            position.y() *= invWeightSum;
          } else if (isScintillator) {
            float x;
            float y;
            etaPhiZToXY(recHits[maxHit].dim1(), recHits[maxHit].dim2(), recHits[maxHit].dim3(), x, y);
            position.x() = x;
            position.y() = y;
          } else {
            position.x() = recHits[maxHit].dim1();
            position.y() = recHits[maxHit].dim2();
          }
          position.z() = recHits[maxHit].dim3();
          position.layer() = recHits[maxHit].layer();

          const uint32_t first = timedOffsets[c];
          const LayerClusterTime time =
              fixSizeHighestDensity(times + first, weights + first, timedOffsets[c + 1] - first, nHitsTime);
          layerClusters.timing()[c].time() = time.time;
          layerClusters.timing()[c].timeError() = time.error;

          layerClusters.energy()[c].correctedEnergy() = -1.f;
          layerClusters.energy()[c].correctedEnergyUncertainty() = -1.f;
          layerClusters.indexes()[c].algoID() = ::reco::CaloCluster::undefined;
          layerClusters.indexes()[c].flags() = 0;
        }
      }
    };

  }  // namespace

  void HGCalLayerClustersSoAAlgoWrapper::run(Queue& queue,
                                             uint32_t nLayerClusters,
                                             float thresholdW0,
                                             float positionDeltaRho2,
                                             bool isScintillator,
                                             uint32_t nHitsTime,
                                             HGCalSoARecHitsDeviceCollection::ConstView recHits,
                                             HGCalSoARecHitsExtraDeviceCollection::ConstView recHitsClustering,
                                             reco::CaloClusterDeviceCollection::View layerClusters) const {
    if (nLayerClusters == 0) {
      return;
    }
    const uint32_t nRecHits = recHits.metadata().size();

    auto x = make_device_view<float>(queue, layerClusters.position().x());
    alpaka::memset(queue, x, 0x0);
    auto y = make_device_view<float>(queue, layerClusters.position().y());
    alpaka::memset(queue, y, 0x0);
    auto cells = make_device_view<int>(queue, layerClusters.position().cells());
    alpaka::memset(queue, cells, 0x0);
    auto energy = make_device_view<float>(queue, layerClusters.energy().energy());
    alpaka::memset(queue, energy, 0x0);
    // reco::CaloID has a virtual destructor, so it cannot be constructed on the device: zero it instead, which gives
    // the detectors() of a default CaloID
    auto caloID = make_device_view<::reco::CaloID>(queue, layerClusters.indexes().caloID());
    alpaka::memset(queue, caloID, 0x0);
    // per layer cluster: the sum of the position weights, and the key of the most energetic rechit (0 is below any
    // rechit)
    auto positionWeightSum = make_device_buffer<float[]>(queue, nLayerClusters);
    alpaka::memset(queue, positionWeightSum, 0x0);
    auto maxEnergyKey = make_device_buffer<uint64_t[]>(queue, nLayerClusters);
    alpaka::memset(queue, maxEnergyKey, 0x0);
    // The times and time weights of the rechits with a time, grouped by layer cluster: the ones of layer cluster c are
    // at [timedOffsets[c], timedOffsets[c + 1]). Their number is not known on the host, so times and weights are sized
    // for all the rechits.
    auto timedHits = make_device_buffer<uint32_t[]>(queue, nLayerClusters);
    alpaka::memset(queue, timedHits, 0x0);
    auto timedOffsets = make_device_buffer<uint32_t[]>(queue, nLayerClusters + 1);
    alpaka::memset(queue, timedOffsets, 0x0);
    auto timedFilled = make_device_buffer<uint32_t[]>(queue, nLayerClusters);
    alpaka::memset(queue, timedFilled, 0x0);
    auto times = make_device_buffer<float[]>(queue, nRecHits);
    auto weights = make_device_buffer<float[]>(queue, nRecHits);

    constexpr uint32_t blockSize = 64;
    const auto workDivRecHits = make_workdiv<Acc1D>(divide_up_by(nRecHits, blockSize), blockSize);
    const auto workDivLayerClusters = make_workdiv<Acc1D>(divide_up_by(nLayerClusters, blockSize), blockSize);
    alpaka::exec<Acc1D>(queue,
                        workDivRecHits,
                        KernelLayerClusterEnergy{},
                        isScintillator,
                        recHits,
                        recHitsClustering,
                        layerClusters,
                        maxEnergyKey.data(),
                        timedHits.data());
    // inclusive scan of the counts, so timedOffsets[c + 1] is where the segment of layer cluster c ends
    iterativePrefixScan<Acc1D>(timedHits.data(), timedOffsets.data() + 1, nLayerClusters, queue);
    alpaka::exec<Acc1D>(queue,
                        workDivRecHits,
                        KernelLayerClusterPositionSums{},
                        thresholdW0,
                        positionDeltaRho2,
                        isScintillator,
                        recHits,
                        recHitsClustering,
                        layerClusters,
                        positionWeightSum.data(),
                        maxEnergyKey.data(),
                        timedOffsets.data(),
                        timedFilled.data(),
                        times.data(),
                        weights.data());
    alpaka::exec<Acc1D>(queue,
                        workDivLayerClusters,
                        KernelLayerClusterPositionAndTime{},
                        isScintillator,
                        nHitsTime,
                        recHits,
                        layerClusters,
                        positionWeightSum.data(),
                        maxEnergyKey.data(),
                        timedOffsets.data(),
                        times.data(),
                        weights.data());
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
