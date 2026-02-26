#ifndef RecoLocalTracker_SiPixelClusterizer_alpaka_PixelClustering_h
#define RecoLocalTracker_SiPixelClusterizer_alpaka_PixelClustering_h

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "FWCore/Utilities/interface/DeviceGlobal.h"
#include "FWCore/Utilities/interface/HostDeviceConstant.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/SimpleVector.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/warpsize.h"

//#define GPU_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE::pixelClustering {

#ifdef GPU_DEBUG
  DEVICE_GLOBAL uint32_t gMaxHit = 0;
#endif

  namespace pixelStatus {
    // Phase-1 pixel modules
    constexpr uint32_t pixelSizeX = pixelTopology::Phase1::numRowsInModule;  // 2 x 80 = 160
    constexpr uint32_t pixelSizeY = pixelTopology::Phase1::numColsInModule;  // 8 x 52 = 416

    // 2-buffer scheme: 1 bit per pixel in each buffer (image + temp)
    // The pixel status is encoded by the combination of both bits:
    //
    //   image | temp | status
    //   ------|------|------------------
    //     0   |  0   | empty
    //     1   |  0   | true pixel
    //     0   |  1   | fake pixel (after expansion)
    //     1   |  1   | duplicate
    constexpr uint32_t bits = 1;
    constexpr uint32_t mask = 1;
    constexpr uint32_t valuesPerWord = sizeof(uint32_t) * 8 / bits;     // 32 values per 32-bit word
    constexpr uint32_t size = pixelSizeX * pixelSizeY / valuesPerWord;  // 160 x 416 / 32 = 2080 32-bit words
    constexpr uint32_t rowSize = pixelSizeX / valuesPerWord;            // 160 / 32 = 5 words per row

    // 1-bit versions for new 2-buffer scheme
    ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr uint32_t getIndex(uint16_t x, uint16_t y) {
      return (pixelSizeX * y + x) / valuesPerWord;
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE constexpr uint32_t getShift(uint16_t x, uint16_t y) {
      return (x % valuesPerWord) * bits;
    }

    // Record a pixel using the 2-buffer scheme (image + temp, 1-bit each).
    // Returns: 0 if pixel was empty (now found), 1 if pixel was already found (duplicate).
    ALPAKA_FN_ACC ALPAKA_FN_INLINE int promote(
        Acc1D const& acc, uint32_t* image, uint32_t* temp, const uint16_t x, const uint16_t y) {
      uint32_t index = getIndex(x, y);
      uint32_t shift = getShift(x, y);
      uint32_t bit = mask << shift;

      // Try to set the image bit (marks pixel as found)
      uint32_t old_image = alpaka::atomicOr(acc, &image[index], bit, alpaka::hierarchy::Threads{});

      if ((old_image & bit) == 0) {
        // Image bit was 0, now set to 1 -> pixel was empty, now found
        return 0;
      } else {
        // Image bit was already 1 -> pixel was already found, mark as duplicate
        // Set the temp bit to mark as duplicate
        alpaka::atomicOr(acc, &temp[index], bit, alpaka::hierarchy::Threads{});
        return 1;
      }
    }

    // Check if a pixel is duplicate using the new 2-buffer scheme.
    // Duplicate = image bit set AND temp bit set.
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool isDuplicate(uint32_t const* __restrict__ image,
                                                    uint32_t const* __restrict__ temp,
                                                    uint16_t x,
                                                    uint16_t y) {
      uint32_t index = getIndex(x, y);
      uint32_t shift = getShift(x, y);
      return ((image[index] >> shift) & mask) && ((temp[index] >> shift) & mask);
    }

    ALPAKA_FN_ACC
    inline bool isMorphingModule(uint32_t moduleId, const uint32_t* morphingModules, uint32_t nMorphingModules) {
      // Binary search for moduleId in sorted morphingModules
      int left = 0, right = static_cast<int>(nMorphingModules) - 1;
      while (left <= right) {
        int mid = left + (right - left) / 2;
        uint32_t val = morphingModules[mid];
        if (val == moduleId)
          return true;
        if (val < moduleId)
          left = mid + 1;
        else
          right = mid - 1;
      }
      return false;
    }
  }  // namespace pixelStatus

  template <typename TrackerTraits>
  struct CountModules {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SiPixelDigisSoAView digi_view,
                                  SiPixelClustersSoAView clus_view,
                                  const unsigned int numElements) const {
      // Make sure the atomicInc below does not overflow
      static_assert(TrackerTraits::numberOfModules < ::pixelClustering::maxNumModules);

#ifdef GPU_DEBUG
      if (cms::alpakatools::once_per_grid(acc)) {
        printf("Starting to count modules to set module starts:");
      }
#endif
      for (int32_t i : cms::alpakatools::uniform_elements(acc, numElements)) {
        // Initialise each pixel with a cluster id equal to its index
        digi_view[i].clus() = i;
        if (::pixelClustering::invalidModuleId == digi_view[i].moduleId())
          continue;

        int32_t j = i - 1;
        while (j >= 0 and digi_view[j].moduleId() == ::pixelClustering::invalidModuleId)
          --j;
        if (j < 0 or digi_view[j].moduleId() != digi_view[i].moduleId()) {
          // Found a module boundary: count the number of modules in  clus_view[0].moduleStart()
          auto loc = alpaka::atomicInc(acc,
                                       &clus_view[0].moduleStart(),
                                       static_cast<uint32_t>(::pixelClustering::maxNumModules),
                                       alpaka::hierarchy::Blocks{});
          ALPAKA_ASSERT_ACC(loc < TrackerTraits::numberOfModules);
#ifdef GPU_DEBUG
          printf("> New module (no. %d) found at digi %d \n", loc, i);
#endif
          clus_view[loc + 1].moduleStart() = i;
        }
      }
    }
  };

  template <typename TrackerTraits>
  struct FindClus {
    // assume that we can cover the whole module with up to 16 blockDimension-wide iterations

    // this must be larger than maxPixInModule / maxIterClustering, and should be a multiple of the warp size
    static constexpr uint32_t maxElementsPerBlock =
        cms::alpakatools::round_up_by(TrackerTraits::maxPixInModule / TrackerTraits::maxIterClustering, 64);
    static constexpr uint32_t maxElementsPerBlockMorph = cms::alpakatools::round_up_by(
        (TrackerTraits::maxPixInModule + TrackerTraits::maxPixInModuleForMorphing) / TrackerTraits::maxIterClustering,
        64);
    static_assert(maxElementsPerBlockMorph >= maxElementsPerBlock);

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SiPixelDigisSoAView digi_view,
                                  SiPixelDigisSoAView fakes_view,
                                  bool enableDigiMorphing,
                                  uint32_t* morphingModules,
                                  uint32_t nMorphingModules,
                                  uint32_t maxFakesInModule,
                                  SiPixelClustersSoAView clus_view,
                                  const unsigned int numElements) const {
      static_assert(TrackerTraits::numberOfModules < ::pixelClustering::maxNumModules);

      auto& lastPixel = alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
      auto& fakePixels = alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
#ifdef GPU_DEBUG
      auto& goodPixels = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
#endif

      const uint32_t lastModule = clus_view[0].moduleStart();
      for (uint32_t module : cms::alpakatools::independent_groups(acc, lastModule)) {
        auto block = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];

        auto firstPixel = clus_view[1 + module].moduleStart();
        uint32_t thisModuleId = digi_view[firstPixel].moduleId();
        uint32_t rawModuleId = digi_view[firstPixel].rawIdArr();
        bool applyDigiMorphing =
            enableDigiMorphing && pixelStatus::isMorphingModule(rawModuleId, morphingModules, nMorphingModules);
        ALPAKA_ASSERT_ACC(thisModuleId < TrackerTraits::numberOfModules);

#ifdef GPU_DEBUG
        if (thisModuleId % 100 == 1)
          if (cms::alpakatools::once_per_block(acc))
            printf("start clusterizer for module %4d in block %4d\n", thisModuleId, block);
#endif

        // Find the index of the first pixel not belonging to this module (or invalid).
        // Note: modules are not consecutive in clus_view, so we cannot use something like
        // lastPixel = (module + 1 == lastModule) ? numElements : clus_view[2 + module].moduleStart();
        lastPixel = numElements;
        const uint32_t firstFake = maxFakesInModule * block;
        fakePixels = 0;
#ifdef GPU_DEBUG
        goodPixels = 0;
#endif
        alpaka::syncBlockThreads(acc);

        for (uint32_t i : cms::alpakatools::independent_group_elements(acc, firstPixel, numElements)) {
          auto id = digi_view[i].moduleId();
          // skip invalid pixels
          if (id == ::pixelClustering::invalidModuleId)
            continue;
          // find the first pixel in a different module
          if (id != thisModuleId) {
            alpaka::atomicMin(acc, &lastPixel, i, alpaka::hierarchy::Threads{});
            break;
          }
        }

        // clear the fake pixels
        if (applyDigiMorphing) {
          // Assume no more than `maxFakesInModule` fake (recovered) pixels per module.
          // The fake pixels for the `module`-th module are stored in `fakes_view[...]` starting at index `firstFake`,
          // equal to `maxFakesInModule * module`.
          ALPAKA_ASSERT_ACC(static_cast<unsigned int>(fakes_view.metadata().size()) >= firstFake + maxFakesInModule);
          for (uint32_t i :
               cms::alpakatools::independent_group_elements(acc, firstFake, firstFake + maxFakesInModule)) {
            // The cluster id of the fake pixels do not need to be unique across modules, as they are transient and only
            // using within each module. They can start independently at `numElements` on each module.
            // This ensures that when comparing the cluser id of a valid pixel with a fake pixel (with `atomicMin`) the
            // resulting cluster id will be the one of the valid pixel.
            fakes_view[i] = {static_cast<int32_t>(numElements + i),  // cluster id
                             0,                                      // pdigi
                             0,                                      // rawIdArr
                             0,                                      // rawADC
                             0,                                      // adc
                             0,                                      // xx
                             0,                                      // yy
                             ::pixelClustering::invalidModuleId};    // moduleId
          }
        }

        using Hist =
            cms::alpakatools::HistoContainer<uint16_t,
                                             TrackerTraits::clusterBinning,
                                             TrackerTraits::maxPixInModule + TrackerTraits::maxPixInModuleForMorphing,
                                             TrackerTraits::clusterBits,
                                             uint16_t>;
        constexpr int warpSize = cms::alpakatools::warpSize;
        auto& hist = alpaka::declareSharedVar<Hist, __COUNTER__>(acc);
        auto& ws = alpaka::declareSharedVar<typename Hist::Counter[warpSize], __COUNTER__>(acc);
        for (uint32_t j : cms::alpakatools::independent_group_elements(acc, Hist::totbins())) {
          hist.off[j] = 0;
          hist.content[j] = 0;
        }
        alpaka::syncBlockThreads(acc);

        ALPAKA_ASSERT_ACC((lastPixel == numElements) or
                          ((lastPixel < numElements) and (digi_view[lastPixel].moduleId() != thisModuleId)));
        // Limit the number of pixels to to maxPixInModule.
        // FIXME if this happens frequently (and not only in simulation with low threshold) one will need to implement something cleverer.
        if (cms::alpakatools::once_per_block(acc)) {
          if (lastPixel - firstPixel > TrackerTraits::maxPixInModule) {
            printf("Too many pixels in module %u: %u > %u\n",
                   thisModuleId,
                   lastPixel - firstPixel,
                   TrackerTraits::maxPixInModule);
            lastPixel = TrackerTraits::maxPixInModule + firstPixel;
          }
        }
        alpaka::syncBlockThreads(acc);
        ALPAKA_ASSERT_ACC(lastPixel - firstPixel <= TrackerTraits::maxPixInModule);

        // remove duplicate pixels
        constexpr bool isPhase2 = std::is_base_of<pixelTopology::Phase2, TrackerTraits>::value;
        if constexpr (not isPhase2) {
          // New 2-buffer scheme: 1-bit per pixel (32 pixels per 32-bit word)
          // image: 0=empty, 1=found/duplicate
          // temp:  0=empty/found, 1=fake/duplicate
          auto& image = alpaka::declareSharedVar<uint32_t[pixelStatus::size], __COUNTER__>(acc);
          auto& temp = alpaka::declareSharedVar<uint32_t[pixelStatus::size], __COUNTER__>(acc);

          if (lastPixel > 1) {
            for (uint32_t i : cms::alpakatools::independent_group_elements(acc, pixelStatus::size)) {
              image[i] = 0;
              temp[i] = 0;
            }
            alpaka::syncBlockThreads(acc);

            for (uint32_t i : cms::alpakatools::independent_group_elements(acc, firstPixel, lastPixel)) {
              // skip invalid pixels
              if (digi_view[i].moduleId() == ::pixelClustering::invalidModuleId)
                continue;
              pixelStatus::promote(acc, image, temp, digi_view[i].xx(), digi_view[i].yy());
            }
            alpaka::syncBlockThreads(acc);

            for (uint32_t i : cms::alpakatools::independent_group_elements(acc, firstPixel, lastPixel)) {
              // skip invalid pixels
              if (digi_view[i].moduleId() == ::pixelClustering::invalidModuleId)
                continue;
              // Duplicate = image bit AND temp bit both set
              if (pixelStatus::isDuplicate(image, temp, digi_view[i].xx(), digi_view[i].yy())) {
                // Mark all duplicate pixels as invalid.
                // Note: the alternative approach to keep a single one of the duplicates would probably make more sense.
                // According to Danek (16 March 2022): "The best would be to ignore such hits, most likely they are just
                // noise. Accepting just one is also OK, any of them."
                digi_view[i].moduleId() = ::pixelClustering::invalidModuleId;
                digi_view[i].rawIdArr() = 0;
              }
            }
            alpaka::syncBlockThreads(acc);

            // apply the digi morphing recovery algorithm
            if (applyDigiMorphing) {
              using namespace pixelStatus;

              // Mark all duplicate pixels as empty in the image, to let the morphing attempt to recover them.
              // In the new 2-buffer scheme: duplicates have image=1, temp=1
              // Clear both bits to make them empty (image=0, temp=0)
              for (uint32_t i : cms::alpakatools::independent_group_elements(acc, size)) {
                // Duplicates are where both image and temp bits are set
                uint32_t duplicates = image[i] & temp[i];
                image[i] &= ~duplicates;
                temp[i] &= ~duplicates;
              }
              alpaka::syncBlockThreads(acc);

              // use the image buffer as a 1-bit-per-pixel image, with 32 pixels packed in each 32-bit word
              // ......  ...............................................................  .....
              // .....#  #.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#  #....
              // .....# [#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#] #....
              // .....#  #.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#  #....
              // ......  ...............................................................  .....

              // first step: expand - read from image, write to temp (where image is 0)
              // Mark empty pixels (image=0) as fake (temp=1) if any neighbor has image=1
              for (uint32_t i : cms::alpakatools::independent_group_elements(acc, size)) {
                uint16_t x = i % rowSize * valuesPerWord;  // 0..4 x 32 = 0, 32, 64, 96, 128
                uint16_t y = i / rowSize;                  // 0..2079 / 5 = 0..415
                uint32_t img = image[i];

                // Build 64-bit buffer: OR together current, above, below rows
                // Shifted by 1 to make room for left edge pixel at bit 0
                uint64_t buffer = static_cast<uint64_t>(img) << 1;
                if (y > 0) {
                  buffer |= static_cast<uint64_t>(image[i - rowSize]) << 1;
                }
                if (y < pixelSizeY - 1) {
                  buffer |= static_cast<uint64_t>(image[i + rowSize]) << 1;
                }

                // Add left edge pixel (bit 31 from previous word -> bit 0 of buffer)
                if (x > 0) {
                  buffer |= static_cast<uint64_t>(image[i - 1] >> 31) & mask;
                  if (y > 0)
                    buffer |= static_cast<uint64_t>(image[i - rowSize - 1] >> 31) & mask;
                  if (y < pixelSizeY - 1)
                    buffer |= static_cast<uint64_t>(image[i + rowSize - 1] >> 31) & mask;
                }

                // Add right edge pixel (bit 0 from next word -> bit 33 of buffer)
                if (x < pixelSizeX - valuesPerWord) {
                  buffer |= static_cast<uint64_t>(image[i + 1] & mask) << 33;
                  if (y > 0)
                    buffer |= static_cast<uint64_t>(image[i - rowSize + 1] & mask) << 33;
                  if (y < pixelSizeY - 1)
                    buffer |= static_cast<uint64_t>(image[i + rowSize + 1] & mask) << 33;
                }

                // For each pixel where image=0, check if any neighbor (in merged buffer) is set
                // OR together 3 shifted versions of buffer to check all neighbors at once: buffer[j] | buffer[j+1] | buffer[j+2] for each bit position j
                uint32_t neighbors = static_cast<uint32_t>(buffer) | static_cast<uint32_t>(buffer >> 1) |
                                     static_cast<uint32_t>(buffer >> 2);
                // Mark as fake only where image=0 (empty) AND neighbors exist
                uint32_t fake = neighbors & ~img;
                // write fake pixels to temp buffer
                temp[i] |= fake;
              }
              alpaka::syncBlockThreads(acc);

              // second step: erode, and create new fake pixels for the remaining ones
              // Read from image | temp to check non-empty neighbors (image=1 OR temp=1)
              // Process only fake pixels (image=0, temp=1)
              for (uint32_t i : cms::alpakatools::independent_group_elements(acc, size)) {
                uint16_t x = i % rowSize * valuesPerWord;  // 0..4 x 32 = 0, 32, 64, 96, 128
                uint16_t y = i / rowSize;                  // 0..2079 / 5 = 0..415
                uint32_t img = image[i];
                uint32_t tmp = temp[i];
                uint32_t value = img | tmp;  // non-empty = image OR temp
                uint32_t above = (y > 0) ? (image[i - rowSize] | temp[i - rowSize]) : 0;
                uint32_t below = (y < pixelSizeY - 1) ? (image[i + rowSize] | temp[i + rowSize]) : 0;
                // First pixel (j = 0)
                {
                  // shift = 0
                  // Process only fake (recovered) pixels: image=0, temp=1
                  if ((img & mask) == 0 and (tmp & mask) != 0) {
                    // If there are no pixels on the edge, pretend it is a fake (recovered) one.
                    uint32_t edge = (x > 0) ? ((image[i - 1] | temp[i - 1]) >> (valuesPerWord - 1)) & mask : 1;
                    // Check that the pixels to the left, above, below, and to the right are not empty.
                    if (edge != 0 and (above & mask) != 0 and (below & mask) != 0 and ((value >> bits) & mask) != 0) {
                      // Create a fake pixel, up to maxFakesInModule pixels per module.
                      unsigned int index =
                          alpaka::atomicInc(acc, &fakePixels, 0xffffffff, alpaka::hierarchy::Threads{});
                      if (index < maxFakesInModule) {
                        auto fake = fakes_view[firstFake + index];
                        ALPAKA_ASSERT_ACC(fake.clus() == static_cast<int32_t>(numElements + firstFake + index));
                        fake.xx() = x;
                        fake.yy() = y;
                        fake.moduleId() = thisModuleId;
                      } else {
                        printf("Too many pixels recovered by digi morphing in module %u: %u > %u\n",
                               thisModuleId,
                               index,
                               maxFakesInModule);
                      }
                    }
                  }
                }
                // Non-edge pixels (j = 1..30)
                for (uint32_t j = 1; j < valuesPerWord - 1; ++j) {
                  uint32_t shift = j * bits;
                  // Process only fake (recovered) pixels: image=0, temp=1
                  if (((img >> shift) & mask) == 0 and ((tmp >> shift) & mask) != 0) {
                    // Check that the pixels to the left, above, below, and to the right are not empty.
                    if (((value >> (shift - bits)) & mask) != 0 and ((above >> shift) & mask) != 0 and
                        ((below >> shift) & mask) != 0 and ((value >> (shift + bits)) & mask) != 0) {
                      // Create a fake pixel, up to maxFakesInModule pixels per module.
                      unsigned int index =
                          alpaka::atomicInc(acc, &fakePixels, 0xffffffff, alpaka::hierarchy::Threads{});
                      if (index < maxFakesInModule) {
                        auto fake = fakes_view[firstFake + index];
                        ALPAKA_ASSERT_ACC(fake.clus() == static_cast<int32_t>(numElements + firstFake + index));
                        fake.xx() = x + j;
                        fake.yy() = y;
                        fake.moduleId() = thisModuleId;
                      } else {
                        printf("Too many pixels recovered by digi morphing in module %u: %u > %u\n",
                               thisModuleId,
                               index,
                               maxFakesInModule);
                      }
                    }
                  }
                }
                // Last pixel (j = 31)
                {
                  uint32_t shift = (valuesPerWord - 1) * bits;
                  // Process only fake (recovered) pixels: image=0, temp=1
                  if (((img >> shift) & mask) == 0 and ((tmp >> shift) & mask) != 0) {
                    // If there are no pixels on the edge, pretend it is a fake (recovered) one.
                    uint32_t edge = (x < pixelSizeX - valuesPerWord) ? ((image[i + 1] | temp[i + 1]) & mask) : 1;
                    // Check that the pixels to the left, above, below, and to the right are not empty.
                    if (((value >> (shift - bits)) & mask) != 0 and ((above >> shift) & mask) != 0 and
                        ((below >> shift) & mask) != 0 and edge != 0) {
                      // Create a fake pixel, up to maxFakesInModule pixels per module.
                      unsigned int index =
                          alpaka::atomicInc(acc, &fakePixels, 0xffffffff, alpaka::hierarchy::Threads{});
                      if (index < maxFakesInModule) {
                        auto fake = fakes_view[firstFake + index];
                        ALPAKA_ASSERT_ACC(fake.clus() == static_cast<int32_t>(numElements + firstFake + index));
                        fake.xx() = x + valuesPerWord - 1;
                        fake.yy() = y;
                        fake.moduleId() = thisModuleId;
                      } else {
                        printf("Too many pixels recovered by digi morphing in module %u: %u > %u\n",
                               thisModuleId,
                               index,
                               maxFakesInModule);
                      }
                    }
                  }
                }
              }
              alpaka::syncBlockThreads(acc);

              // Clamp fakePixels to maxFakesInModule
              if (cms::alpakatools::once_per_block(acc)) {
                if (fakePixels > maxFakesInModule)
                  fakePixels = maxFakesInModule;
              }
              alpaka::syncBlockThreads(acc);

            }  // if (applyDigiMorphing)
          }  // if (lastPixel > 1)
        }  // if constexpr (not isPhase2)

        // fill histo
        for (uint32_t i : cms::alpakatools::independent_group_elements(acc, firstPixel, lastPixel)) {
          // skip invalid pixels
          if (digi_view[i].moduleId() != ::pixelClustering::invalidModuleId) {
            hist.count(acc, digi_view[i].yy());
#ifdef GPU_DEBUG
            alpaka::atomicAdd(acc, &goodPixels, 1u, alpaka::hierarchy::Threads{});
#endif
          }
        }
        if (applyDigiMorphing) {
          for (uint32_t i : cms::alpakatools::independent_group_elements(acc, firstFake, firstFake + fakePixels)) {
            hist.count(acc, fakes_view[i].yy());
          }
        }
        for (uint32_t i : cms::alpakatools::independent_group_elements(acc, warpSize)) {
          ws[i] = 0;  // used by prefix scan...
        }
        alpaka::syncBlockThreads(acc);
        hist.finalize(acc, ws);
        alpaka::syncBlockThreads(acc);

#ifdef GPU_DEBUG
        ALPAKA_ASSERT_ACC(hist.size() == goodPixels + fakePixels);
        if (thisModuleId % 100 == 1) {
          if (cms::alpakatools::once_per_block(acc)) {
            printf(
                "module %d has %d good pixels and recovered %d pixels by morphing\n", module, goodPixels, fakePixels);
            printf("histo size %d\n", hist.size());
          }
        }
#endif
        for (uint32_t i : cms::alpakatools::independent_group_elements(acc, firstPixel, lastPixel)) {
          // skip invalid pixels
          if (digi_view[i].moduleId() != ::pixelClustering::invalidModuleId) {
            // For valid pixels, the id used in the histogram is `i - firstPixel`, ranging from `0` to
            // `lastPixel - firstPixel` (excluded), so in the `[0, TrackerTraits::maxPixInModule)` range.
            hist.fill(acc, digi_view[i].yy(), i - firstPixel);
          }
        }
        if (applyDigiMorphing) {
          // For fake pixels, the id used in the histogram is `i - firstFake + TrackerTraits::maxPixInModule`, so
          // in the range `[TrackerTraits::maxPixInModule, `TrackerTraits::maxPixInModule + maxFakesInModule`).
          // This ensures that the fake pixels have different ids from the valid pixels.
          for (uint32_t i : cms::alpakatools::independent_group_elements(acc, firstFake, firstFake + fakePixels)) {
            hist.fill(acc, fakes_view[i].yy(), i - firstFake + TrackerTraits::maxPixInModule);
          }
        }

#ifdef GPU_DEBUG
        // look for anomalous high occupancy
        auto& n40 = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
        auto& n60 = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
        if (cms::alpakatools::once_per_block(acc)) {
          n40 = 0;
          n60 = 0;
        }
        alpaka::syncBlockThreads(acc);
        for (uint32_t j : cms::alpakatools::independent_group_elements(acc, Hist::nbins())) {
          if (hist.size(j) > 60)
            alpaka::atomicAdd(acc, &n60, 1u, alpaka::hierarchy::Threads{});
          if (hist.size(j) > 40)
            alpaka::atomicAdd(acc, &n40, 1u, alpaka::hierarchy::Threads{});
        }
        alpaka::syncBlockThreads(acc);
        if (cms::alpakatools::once_per_block(acc)) {
          if (n60 > 0)
            printf("columns with more than 60 px %d in %d\n", n60, thisModuleId);
          else if (n40 > 0)
            printf("columns with more than 40 px %d in %d\n", n40, thisModuleId);
        }
        alpaka::syncBlockThreads(acc);
#endif

        [[maybe_unused]] const uint32_t blockDimension = alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u];
        // assume that we can cover the whole module with up to maxIterClustering blockDimension-wide iterations
        ALPAKA_ASSERT_ACC((hist.size() / blockDimension) < TrackerTraits::maxIterClustering);

        // number of elements per thread
        const uint32_t maxElements = cms::alpakatools::requires_single_thread_per_block_v<Acc1D>
                                         ? (enableDigiMorphing ? maxElementsPerBlockMorph : maxElementsPerBlock)
                                         : 1;

#ifdef GPU_DEBUG
        const auto nElementsPerThread = alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u];
        if (nElementsPerThread > maxElements)
          printf("This is WRONG: nElementsPerThread > maxElements : %d > %d\n", nElementsPerThread, maxElements);
        else if (thisModuleId % 500 == 1)
          printf("This is OK: nElementsPerThread <= maxElements : %d <= %d\n", nElementsPerThread, maxElements);
#endif

        ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u] <= maxElements));

        const unsigned int maxIter = TrackerTraits::maxIterClustering * maxElements;

        // nearest neighbours (nn)
        constexpr int maxNeighbours = 8;
        uint16_t nn[maxIter][maxNeighbours];
        uint8_t nnn[maxIter];  // number of nn
        for (uint32_t k = 0; k < maxIter; ++k) {
          nnn[k] = 0;
          for (int j = 0; j < maxNeighbours; ++j) {
            nn[k][j] = 0;
          }
        }

        alpaka::syncBlockThreads(acc);  // for hit filling!

        // fill the nearest neighbours
        uint32_t k = 0;
        for (uint32_t j : cms::alpakatools::independent_group_elements(acc, hist.size())) {
          ALPAKA_ASSERT_ACC(k < maxIter);
          auto p = hist.begin() + j;
          bool isFake = (*p >= TrackerTraits::maxPixInModule);
          uint32_t i;
          uint16_t ix, iy;
          const uint16_t* end;
          if (applyDigiMorphing and isFake) {
            // recovered "fake" pixel
            i = *p - TrackerTraits::maxPixInModule + firstFake;
            auto pixel = fakes_view[i];
            ALPAKA_ASSERT_ACC(pixel.moduleId() != ::pixelClustering::invalidModuleId);
            ALPAKA_ASSERT_ACC(pixel.moduleId() == thisModuleId);  // same module
            ix = pixel.xx();
            iy = pixel.yy();
            auto bin = Hist::bin(iy + 1);
            end = hist.end(bin);
          } else {
            // real pixel
            i = *p + firstPixel;
            auto pixel = digi_view[i];
            ALPAKA_ASSERT_ACC(pixel.moduleId() != ::pixelClustering::invalidModuleId);
            ALPAKA_ASSERT_ACC(pixel.moduleId() == thisModuleId);  // same module
            ix = pixel.xx();
            iy = pixel.yy();
            auto bin = Hist::bin(iy + 1);
            end = hist.end(bin);
          }
          ++p;
          ALPAKA_ASSERT_ACC(0 == nnn[k]);
          for (; p < end; ++p) {
            bool otherIsFake = (*p >= TrackerTraits::maxPixInModule);
            uint32_t m;
            uint16_t mx, my;
            if (applyDigiMorphing and otherIsFake) {
              m = *p - TrackerTraits::maxPixInModule + firstFake;
              auto pixel = fakes_view[m];
              mx = pixel.xx();
              my = pixel.yy();
            } else {
              m = *p + firstPixel;
              auto pixel = digi_view[m];
              mx = pixel.xx();
              my = pixel.yy();
            }
            ALPAKA_ASSERT_ACC(m != i or otherIsFake != isFake);
            ALPAKA_ASSERT_ACC(int(my) - int(iy) >= 0);
            ALPAKA_ASSERT_ACC(int(my) - int(iy) <= 1);
            if (std::abs(int(mx) - int(ix)) <= 1) {
              auto l = nnn[k]++;
              ALPAKA_ASSERT_ACC(l < maxNeighbours);
              nn[k][l] = *p;
            }
          }
          ++k;
        }

        // For each pixel, look at all the pixels until the end of the module;
        // when two valid pixels within +/- 1 in x or y are found, set their id to the minimum;
        // after the loop, all the pixel in each cluster should have the id equeal to the lowest
        // pixel in the cluster ( clus[i] == i ).
        bool done = false;
        while (alpaka::syncBlockThreadsPredicate<alpaka::BlockOr>(acc, not done)) {
          done = true;
          uint32_t k = 0;
          for (uint32_t j : cms::alpakatools::independent_group_elements(acc, hist.size())) {
            ALPAKA_ASSERT_ACC(k < maxIter);
            auto p = hist.begin() + j;
            bool isFake = (*p >= TrackerTraits::maxPixInModule);
            uint32_t i;
            int32_t* iclus;
            if (applyDigiMorphing and isFake) {
              // recovered "fake" pixel
              i = *p - TrackerTraits::maxPixInModule + firstFake;
              iclus = &fakes_view[i].clus();
            } else {
              // real pixel
              i = *p + firstPixel;
              iclus = &digi_view[i].clus();
            }
            for (int kk = 0; kk < nnn[k]; ++kk) {
              auto l = nn[k][kk];
              bool otherIsFake = (l >= TrackerTraits::maxPixInModule);
              uint32_t m;
              int32_t* mclus;
              if (applyDigiMorphing and otherIsFake) {
                // recovered "fake" pixel
                m = l - TrackerTraits::maxPixInModule + firstFake;
                mclus = &fakes_view[m].clus();
              } else {
                // real pixel
                m = l + firstPixel;
                mclus = &digi_view[m].clus();
              }
              ALPAKA_ASSERT_ACC(m != i or otherIsFake != isFake);
              // the algorithm processes one module per block, so the atomic operation's scope is "Threads" (all threads in the current block)
              auto old = alpaka::atomicMin(acc, mclus, *iclus, alpaka::hierarchy::Threads{});
              if (old != *iclus) {
                // end the loop only if no changes were applied
                done = false;
              }
              alpaka::atomicMin(acc, iclus, old, alpaka::hierarchy::Threads{});
            }  // neighbours loop
            ++k;
          }  // pixel loop
          alpaka::syncBlockThreads(acc);
          for (uint32_t j : cms::alpakatools::independent_group_elements(acc, hist.size())) {
            auto p = hist.begin() + j;
            bool isFake = (*p >= TrackerTraits::maxPixInModule);
            uint32_t i, m;
            if (applyDigiMorphing and isFake) {
              // recovered "fake" pixel
              i = *p - TrackerTraits::maxPixInModule + firstFake;
              m = fakes_view[i].clus();
            } else {
              // real pixel
              i = *p + firstPixel;
              m = digi_view[i].clus();
            }

            while (true) {
              uint32_t n;
              if (m < numElements) {
                n = digi_view[m].clus();
              } else {
                n = fakes_view[m - numElements].clus();
              }
              if (m == n) {
                break;
              }
              m = n;
            }

            if (isFake) {
              fakes_view[i].clus() = m;
            } else {
              digi_view[i].clus() = m;
            }
          }
        }  // end while

        /*
        // check that all threads in the block have executed the same number of iterations
        auto& n0 = alpaka::declareSharedVar<int, __COUNTER__>(acc);
        if (cms::alpakatools::once_per_block(acc))
          n0 = nloops;
        alpaka::syncBlockThreads(acc);
        ALPAKA_ASSERT_ACC(alpaka::syncBlockThreadsPredicate<alpaka::BlockAnd>(acc, nloops == n0));
        if (thisModuleId % 100 == 1)
          if (cms::alpakatools::once_per_block(acc))
            printf("# loops %d\n", nloops);
        */

        auto& foundClusters = alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
        foundClusters = 0;
        alpaka::syncBlockThreads(acc);

        // find the number of different clusters, identified by a pixels with clus[i] == i;
        // mark these pixels with a negative id.
        for (uint32_t i : cms::alpakatools::independent_group_elements(acc, firstPixel, lastPixel)) {
          // skip invalid pixels
          if (digi_view[i].moduleId() == ::pixelClustering::invalidModuleId)
            continue;
          if (digi_view[i].clus() == static_cast<int>(i)) {
            auto old = alpaka::atomicInc(acc, &foundClusters, 0xffffffff, alpaka::hierarchy::Threads{});
            digi_view[i].clus() = -(old + 1);
          }
        }
        alpaka::syncBlockThreads(acc);

        // propagate the negative id to all the pixels in the cluster.
        for (uint32_t i : cms::alpakatools::independent_group_elements(acc, firstPixel, lastPixel)) {
          // skip invalid pixels
          if (digi_view[i].moduleId() == ::pixelClustering::invalidModuleId)
            continue;
          if (digi_view[i].clus() >= 0) {
            // mark each pixel in a cluster with the same id as the first one
            digi_view[i].clus() = digi_view[digi_view[i].clus()].clus();
          }
        }
        alpaka::syncBlockThreads(acc);

        // adjust the cluster id to be a positive value starting from 0
        for (uint32_t i : cms::alpakatools::independent_group_elements(acc, firstPixel, lastPixel)) {
          if (digi_view[i].moduleId() == ::pixelClustering::invalidModuleId) {
            // mark invalid pixels with an invalid cluster index
            digi_view[i].clus() = ::pixelClustering::invalidClusterId;
          } else {
            digi_view[i].clus() = -digi_view[i].clus() - 1;
          }
        }
        alpaka::syncBlockThreads(acc);

        if (cms::alpakatools::once_per_block(acc)) {
          clus_view[thisModuleId].clusInModule() = foundClusters;
          clus_view[module].moduleId() = thisModuleId;
#ifdef GPU_DEBUG
          if (foundClusters > gMaxHit) {
            gMaxHit = foundClusters;
            if (foundClusters > 8)
              printf("max hit %d in %d\n", foundClusters, thisModuleId);
          }
          if (thisModuleId % 100 == 1)
            printf("%d clusters in module %d\n", foundClusters, thisModuleId);
#endif
        }
      }  // module loop
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::pixelClustering

#endif  // plugin_SiPixelClusterizer_alpaka_PixelClustering.h
