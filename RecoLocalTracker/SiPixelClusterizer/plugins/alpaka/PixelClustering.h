#ifndef RecoLocalTracker_SiPixelClusterizer_alpaka_PixelClustering_h
#define RecoLocalTracker_SiPixelClusterizer_alpaka_PixelClustering_h

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/SimpleVector.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

//#define GPU_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace pixelClustering {

#ifdef GPU_DEBUG
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_STATIC_ACC_MEM_GLOBAL uint32_t gMaxHit = 0;
#endif

    namespace pixelStatus {
      // Phase-1 pixel modules
      constexpr uint32_t pixelSizeX = pixelTopology::Phase1::numRowsInModule;
      constexpr uint32_t pixelSizeY = pixelTopology::Phase1::numColsInModule;

      // Use 0x00, 0x01, 0x03 so each can be OR'ed on top of the previous ones
      enum Status : uint32_t { kEmpty = 0x00, kFound = 0x01, kDuplicate = 0x03 };

      constexpr uint32_t bits = 2;
      constexpr uint32_t mask = (0x01 << bits) - 1;
      constexpr uint32_t valuesPerWord = sizeof(uint32_t) * 8 / bits;
      constexpr uint32_t size = pixelSizeX * pixelSizeY / valuesPerWord;

      ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr uint32_t getIndex(uint16_t x, uint16_t y) {
        return (pixelSizeX * y + x) / valuesPerWord;
      }

      ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr uint32_t getShift(uint16_t x, uint16_t y) {
        return (x % valuesPerWord) * 2;
      }

      ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr Status getStatus(uint32_t const* __restrict__ status,
                                                                uint16_t x,
                                                                uint16_t y) {
        uint32_t index = getIndex(x, y);
        uint32_t shift = getShift(x, y);
        return Status{(status[index] >> shift) & mask};
      }

      ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr bool isDuplicate(uint32_t const* __restrict__ status,
                                                                uint16_t x,
                                                                uint16_t y) {
        return getStatus(status, x, y) == kDuplicate;
      }

      /* FIXME
       * In the more general case (e.g. a multithreaded CPU backend) there is a potential race condition
       * between the read of status[index] at line NNN and the atomicCas at line NNN.
       * We should investigate:
       *   - if `status` should be read through a `volatile` pointer (CUDA/ROCm)
       *   - if `status` should be read with an atomic load (CPU)
       */
      template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr void promote(TAcc const& acc,
                                                            uint32_t* __restrict__ status,
                                                            const uint16_t x,
                                                            const uint16_t y) {
        uint32_t index = getIndex(x, y);
        uint32_t shift = getShift(x, y);
        uint32_t old_word = status[index];
        uint32_t expected = old_word;
        do {
          expected = old_word;
          Status old_status{(old_word >> shift) & mask};
          if (kDuplicate == old_status) {
            // nothing to do
            return;
          }
          Status new_status = (kEmpty == old_status) ? kFound : kDuplicate;
          uint32_t new_word = old_word | (static_cast<uint32_t>(new_status) << shift);
          old_word = alpaka::atomicCas(acc, &status[index], expected, new_word, alpaka::hierarchy::Blocks{});
        } while (expected != old_word);
      }

    }  // namespace pixelStatus

    template <typename TrackerTraits>
    struct CountModules {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    SiPixelDigisSoAView digi_view,
                                    SiPixelClustersSoAView clus_view,
                                    const unsigned int numElements) const {
        [[maybe_unused]] constexpr int nMaxModules = TrackerTraits::numberOfModules;

#ifdef GPU_DEBUG
        if (cms::alpakatools::once_per_grid(acc)) {
          printf("Starting to count modules to set module starts:");
        }
#endif
        cms::alpakatools::for_each_element_in_grid_strided(acc, numElements, [&](uint32_t i) {
          digi_view[i].clus() = i;
          if (::pixelClustering::invalidModuleId != digi_view[i].moduleId()) {
            int j = i - 1;
            while (j >= 0 and digi_view[j].moduleId() == ::pixelClustering::invalidModuleId)
              --j;
            if (j < 0 or digi_view[j].moduleId() != digi_view[i].moduleId()) {
              // boundary...
              auto loc = alpaka::atomicInc(
                  acc, clus_view.moduleStart(), std::decay_t<uint32_t>(nMaxModules), alpaka::hierarchy::Blocks{});
#ifdef GPU_DEBUG
              printf("> New module (no. %d) found at digi %d \n", loc, i);
#endif
              clus_view[loc + 1].moduleStart() = i;
            }
          }
        });
      }
    };

    template <typename TrackerTraits>
    struct FindClus {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    SiPixelDigisSoAView digi_view,
                                    SiPixelClustersSoAView clus_view,
                                    const unsigned int numElements) const {
        constexpr bool isPhase2 = std::is_base_of<pixelTopology::Phase2, TrackerTraits>::value;
        constexpr const uint32_t pixelStatusSize = isPhase2 ? 1 : pixelStatus::size;

        // packed words array used to store the pixelStatus of each pixel
        auto& status = alpaka::declareSharedVar<uint32_t[pixelStatusSize], __COUNTER__>(acc);

        // find the index of the first pixel not belonging to this module (or invalid)
        auto& msize = alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);

        const uint32_t blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
        if (blockIdx >= clus_view[0].moduleStart())
          return;

        auto firstModule = blockIdx;
        auto endModule = clus_view[0].moduleStart();

        const uint32_t gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u]);

        for (auto module = firstModule; module < endModule; module += gridDimension) {
          auto firstPixel = clus_view[1 + module].moduleStart();
          auto thisModuleId = digi_view[firstPixel].moduleId();
          ALPAKA_ASSERT_OFFLOAD(thisModuleId < TrackerTraits::numberOfModules);
#ifdef GPU_DEBUG
          if (thisModuleId % 100 == 1)
            if (cms::alpakatools::once_per_block(acc))
              printf("start clusterizer for module %d in block %d\n", thisModuleId, module);
#endif

          msize = numElements;
          alpaka::syncBlockThreads(acc);

          // Stride = block size.
          const uint32_t blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);

          // Get thread / CPU element indices in block.
          const auto& [firstElementIdxNoStride, endElementIdxNoStride] =
              cms::alpakatools::element_index_range_in_block(acc, firstPixel);
          uint32_t firstElementIdx = firstElementIdxNoStride;
          uint32_t endElementIdx = endElementIdxNoStride;

          // skip threads not associated to an existing pixel
          for (uint32_t i = firstElementIdx; i < numElements; ++i) {
            if (not cms::alpakatools::next_valid_element_index_strided(
                    i, firstElementIdx, endElementIdx, blockDimension, numElements))
              break;
            auto id = digi_view[i].moduleId();
            if (id == ::pixelClustering::invalidModuleId)  // skip invalid pixels
              continue;
            if (id != thisModuleId) {  // find the first pixel in a different module
              alpaka::atomicMin(acc, &msize, i, alpaka::hierarchy::Threads{});
              break;
            }
          }
          //init hist  (ymax=416 < 512 : 9bits)
          constexpr uint32_t maxPixInModule = TrackerTraits::maxPixInModule;
          constexpr auto nbins = TrackerTraits::clusterBinning;
          constexpr auto nbits = TrackerTraits::clusterBits;
          using Hist = cms::alpakatools::HistoContainer<uint16_t, nbins, maxPixInModule, nbits, uint16_t>;
          auto& hist = alpaka::declareSharedVar<Hist, __COUNTER__>(acc);
          auto& ws = alpaka::declareSharedVar<typename Hist::Counter[32], __COUNTER__>(acc);
          cms::alpakatools::for_each_element_in_block_strided(
              acc, Hist::totbins(), [&](uint32_t j) { hist.off[j] = 0; });
          alpaka::syncBlockThreads(acc);
          ALPAKA_ASSERT_OFFLOAD((msize == numElements) or
                                ((msize < numElements) and (digi_view[msize].moduleId() != thisModuleId)));
          // limit to maxPixInModule  (FIXME if recurrent (and not limited to simulation with low threshold) one will need to implement something cleverer)
          if (cms::alpakatools::once_per_grid(acc)) {
            if (msize - firstPixel > maxPixInModule) {
              printf("too many pixels in module %d: %d > %d\n", thisModuleId, msize - firstPixel, maxPixInModule);
              msize = maxPixInModule + firstPixel;
            }
          }
          alpaka::syncBlockThreads(acc);
          ALPAKA_ASSERT_OFFLOAD(msize - firstPixel <= maxPixInModule);

#ifdef GPU_DEBUG
          auto& totGood = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
          totGood = 0;
          alpaka::syncBlockThreads(acc);
#endif
          // remove duplicate pixels
          if constexpr (not isPhase2) {  //FIXME remove THIS
            if (msize > 1) {
              cms::alpakatools::for_each_element_in_block_strided(
                  acc, pixelStatus::size, [&](uint32_t i) { status[i] = 0; });
              alpaka::syncBlockThreads(acc);

              cms::alpakatools::for_each_element_in_block_strided(acc, msize - 1, firstElementIdx, [&](uint32_t i) {
                // skip invalid pixels
                if (digi_view[i].moduleId() == ::pixelClustering::invalidModuleId)
                  return;
                pixelStatus::promote(acc, status, digi_view[i].xx(), digi_view[i].yy());
              });
              alpaka::syncBlockThreads(acc);
              cms::alpakatools::for_each_element_in_block_strided(acc, msize - 1, firstElementIdx, [&](uint32_t i) {
                // skip invalid pixels
                if (digi_view[i].moduleId() == ::pixelClustering::invalidModuleId)
                  return;
                if (pixelStatus::isDuplicate(status, digi_view[i].xx(), digi_view[i].yy())) {
                  digi_view[i].moduleId() = ::pixelClustering::invalidModuleId;
                  digi_view[i].rawIdArr() = 0;
                }
              });
              alpaka::syncBlockThreads(acc);
            }
          }
          // fill histo
          cms::alpakatools::for_each_element_in_block_strided(acc, msize, firstPixel, [&](uint32_t i) {
            if (digi_view[i].moduleId() != ::pixelClustering::invalidModuleId) {  // skip invalid pixels
              hist.count(acc, digi_view[i].yy());
#ifdef GPU_DEBUG
              alpaka::atomicAdd(acc, &totGood, 1u, alpaka::hierarchy::Blocks{});
#endif
            }
          });
          alpaka::syncBlockThreads(acc);
          cms::alpakatools::for_each_element_in_block(acc, 32u, [&](uint32_t i) {
            ws[i] = 0;  // used by prefix scan...
          });
          alpaka::syncBlockThreads(acc);
          hist.finalize(acc, ws);
          alpaka::syncBlockThreads(acc);
#ifdef GPU_DEBUG
          ALPAKA_ASSERT_OFFLOAD(hist.size() == totGood);
          if (thisModuleId % 100 == 1)
            if (cms::alpakatools::once_per_block(acc))
              printf("histo size %d\n", hist.size());
#endif
          cms::alpakatools::for_each_element_in_block_strided(acc, msize, firstPixel, [&](uint32_t i) {
            if (digi_view[i].moduleId() != ::pixelClustering::invalidModuleId) {  // skip invalid pixels
              hist.fill(acc, digi_view[i].yy(), i - firstPixel);
            }
          });
          // Assume that we can cover the whole module with up to 16 blockDimension-wide iterations
          // This maxiter value was tuned for GPU, with 256 or 512 threads per block.
          // Hence, also works for CPU case, with 256 or 512 elements per thread.
          // Real constrainst is maxiter = hist.size() / blockDimension,
          // with blockDimension = threadPerBlock * elementsPerThread.
          // Hence, maxiter can be tuned accordingly to the workdiv.
          constexpr unsigned int maxiter = 16;
          ALPAKA_ASSERT_OFFLOAD((hist.size() / blockDimension) <= maxiter);

          // NB: can be tuned.
          constexpr uint32_t threadDimension = cms::alpakatools::requires_single_thread_per_block_v<TAcc> ? 256 : 1;

#ifndef NDEBUG
          [[maybe_unused]] const uint32_t runTimeThreadDimension =
              alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u];
          ALPAKA_ASSERT_OFFLOAD(runTimeThreadDimension <= threadDimension);
#endif

          // nearest neighbour
          // allocate space for duplicate pixels: a pixel can appear more than once with different charge in the same event
          constexpr int maxNeighbours = 10;
          uint16_t nn[maxiter][threadDimension][maxNeighbours];
          uint8_t nnn[maxiter][threadDimension];  // number of nn
          for (uint32_t elementIdx = 0; elementIdx < threadDimension; ++elementIdx) {
            for (uint32_t k = 0; k < maxiter; ++k) {
              nnn[k][elementIdx] = 0;
            }
          }

          alpaka::syncBlockThreads(acc);  // for hit filling!

#ifdef GPU_DEBUG
          // look for anomalous high occupancy
          auto& n40 = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
          auto& n60 = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
          n40 = n60 = 0;
          alpaka::syncBlockThreads(acc);
          cms::alpakatools::for_each_element_in_block_strided(acc, Hist::nbins(), [&](uint32_t j) {
            if (hist.size(j) > 60)
              alpaka::atomicAdd(acc, &n60, 1u, alpaka::hierarchy::Blocks{});
            if (hist.size(j) > 40)
              alpaka::atomicAdd(acc, &n40, 1u, alpaka::hierarchy::Blocks{});
          });
          alpaka::syncBlockThreads(acc);
          if (cms::alpakatools::once_per_block(acc)) {
            if (n60 > 0)
              printf("columns with more than 60 px %d in %d\n", n60, thisModuleId);
            else if (n40 > 0)
              printf("columns with more than 40 px %d in %d\n", n40, thisModuleId);
          }
          alpaka::syncBlockThreads(acc);
#endif
          // fill NN
          uint32_t k = 0u;
          cms::alpakatools::for_each_element_in_block_strided(acc, hist.size(), [&](uint32_t j) {
            const uint32_t jEquivalentClass = j % threadDimension;
            k = j / blockDimension;
            ALPAKA_ASSERT_OFFLOAD(k < maxiter);
            auto p = hist.begin() + j;
            auto i = *p + firstPixel;
            ALPAKA_ASSERT_OFFLOAD(digi_view[i].moduleId() != ::pixelClustering::invalidModuleId);
            ALPAKA_ASSERT_OFFLOAD(digi_view[i].moduleId() == thisModuleId);  // same module
            int be = Hist::bin(digi_view[i].yy() + 1);
            auto e = hist.end(be);
            ++p;
            ALPAKA_ASSERT_OFFLOAD(0 == nnn[k][jEquivalentClass]);
            for (; p < e; ++p) {
              auto m = (*p) + firstPixel;
              ALPAKA_ASSERT_OFFLOAD(m != i);
              ALPAKA_ASSERT_OFFLOAD(int(digi_view[m].yy()) - int(digi_view[i].yy()) >= 0);
              ALPAKA_ASSERT_OFFLOAD(int(digi_view[m].yy()) - int(digi_view[i].yy()) <= 1);
              if (std::abs(int(digi_view[m].xx()) - int(digi_view[i].xx())) <= 1) {
                auto l = nnn[k][jEquivalentClass]++;
                ALPAKA_ASSERT_OFFLOAD(l < maxNeighbours);
                nn[k][jEquivalentClass][l] = *p;
              }
            }
          });
          // for each pixel, look at all the pixels until the end of the module;
          // when two valid pixels within +/- 1 in x or y are found, set their id to the minimum;
          // after the loop, all the pixel in each cluster should have the id equeal to the lowest
          // pixel in the cluster ( clus[i] == i ).
          bool more = true;
          int nloops = 0;
          while (alpaka::syncBlockThreadsPredicate<alpaka::BlockOr>(acc, more)) {
            if (1 == nloops % 2) {
              cms::alpakatools::for_each_element_in_block_strided(acc, hist.size(), [&](uint32_t j) {
                auto p = hist.begin() + j;
                auto i = *p + firstPixel;
                auto m = digi_view[i].clus();
                while (m != digi_view[m].clus())
                  m = digi_view[m].clus();
                digi_view[i].clus() = m;
              });
            } else {
              more = false;
              uint32_t k = 0u;
              cms::alpakatools::for_each_element_in_block_strided(acc, hist.size(), [&](uint32_t j) {
                k = j / blockDimension;
                const uint32_t jEquivalentClass = j % threadDimension;
                auto p = hist.begin() + j;
                auto i = *p + firstPixel;
                for (int kk = 0; kk < nnn[k][jEquivalentClass]; ++kk) {
                  auto l = nn[k][jEquivalentClass][kk];
                  auto m = l + firstPixel;
                  ALPAKA_ASSERT_OFFLOAD(m != i);
                  auto old =
                      alpaka::atomicMin(acc, &digi_view[m].clus(), digi_view[i].clus(), alpaka::hierarchy::Blocks{});
                  if (old != digi_view[i].clus()) {
                    // end the loop only if no changes were applied
                    more = true;
                  }
                  alpaka::atomicMin(acc, &digi_view[i].clus(), old, alpaka::hierarchy::Blocks{});
                }  // nnloop
              });  // pixel loop
            }
            ++nloops;
          }  // end while
#ifdef GPU_DEBUG
          {
            auto& n0 = alpaka::declareSharedVar<int, __COUNTER__>(acc);
            if (cms::alpakatools::once_per_block(acc))
              n0 = nloops;
            alpaka::syncBlockThreads(acc);
#ifndef NDEBUG
            [[maybe_unused]] auto ok = n0 == nloops;
            ALPAKA_ASSERT_OFFLOAD(alpaka::syncBlockThreadsPredicate<alpaka::BlockAnd>(acc, ok));
#endif
            if (thisModuleId % 100 == 1)
              if (cms::alpakatools::once_per_block(acc))
                printf("# loops %d\n", nloops);
          }
#endif
          auto& foundClusters = alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
          foundClusters = 0;
          alpaka::syncBlockThreads(acc);

          // find the number of different clusters, identified by a pixels with clus[i] == i;
          // mark these pixels with a negative id.
          cms::alpakatools::for_each_element_in_block_strided(acc, msize, firstPixel, [&](uint32_t i) {
            if (digi_view[i].moduleId() != ::pixelClustering::invalidModuleId) {  // skip invalid pixels
              if (digi_view[i].clus() == static_cast<int>(i)) {
                auto old = alpaka::atomicInc(acc, &foundClusters, 0xffffffff, alpaka::hierarchy::Threads{});
                digi_view[i].clus() = -(old + 1);
              }
            }
          });
          alpaka::syncBlockThreads(acc);

          // propagate the negative id to all the pixels in the cluster.
          cms::alpakatools::for_each_element_in_block_strided(acc, msize, firstPixel, [&](uint32_t i) {
            if (digi_view[i].moduleId() != ::pixelClustering::invalidModuleId) {  // skip invalid pixels
              if (digi_view[i].clus() >= 0) {
                // mark each pixel in a cluster with the same id as the first one
                digi_view[i].clus() = digi_view[digi_view[i].clus()].clus();
              }
            }
          });
          alpaka::syncBlockThreads(acc);

          // adjust the cluster id to be a positive value starting from 0
          cms::alpakatools::for_each_element_in_block_strided(acc, msize, firstPixel, [&](uint32_t i) {
            if (digi_view[i].moduleId() == ::pixelClustering::invalidModuleId) {  // skip invalid pixels
              digi_view[i].clus() = ::pixelClustering::invalidClusterId;
            } else {
              digi_view[i].clus() = -digi_view[i].clus() - 1;
            }
          });
          alpaka::syncBlockThreads(acc);
          if (cms::alpakatools::once_per_block(acc)) {
            clus_view[thisModuleId].clusInModule() = foundClusters;
            clus_view[module].moduleId() = thisModuleId;
#ifdef GPU_DEBUG
            if (foundClusters > gMaxHit<TAcc>) {
              gMaxHit<TAcc> = foundClusters;
              if (foundClusters > 8)
                printf("max hit %d in %d\n", foundClusters, thisModuleId);
            }
            // if (thisModuleId % 100 == 1)
            printf("%d clusters in module %d\n", foundClusters, thisModuleId);
#endif
          }
        }  // module loop
      }
    };
  }  // namespace pixelClustering
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif  // plugin_SiPixelClusterizer_alpaka_PixelClustering.h
